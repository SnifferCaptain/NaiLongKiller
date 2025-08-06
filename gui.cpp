// 解决Qt和TBB的emit冲突
#define QT_NO_EMIT

#include "gui.hpp"
#include "NLKiller.hpp"
#include "ytensor.hpp"
#include <QApplication>
#include <QDir>
#include <QStandardPaths>
#include <QTextStream>
#include <QDateTime>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>

// 将QImage转换为YTensor<u_char, 3>
YTensor<u_char, 3> qImageToYTensor(const QImage& qimg) {
    // 确保图像是RGB格式
    QImage rgbImage = qimg.convertToFormat(QImage::Format_RGB888);
    
    int height = rgbImage.height();
    int width = rgbImage.width();
    int channels = 3;
    
    YTensor<u_char, 3> tensor(height, width, channels);
    
    // 复制像素数据
    const uchar* srcData = rgbImage.constBits();
    int bytesPerLine = rgbImage.bytesPerLine();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uchar* pixel = srcData + y * bytesPerLine + x * channels;
            // QImage的RGB顺序和我们需要的一致
            tensor.at(y, x, 0) = pixel[0]; // R
            tensor.at(y, x, 1) = pixel[1]; // G  
            tensor.at(y, x, 2) = pixel[2]; // B
        }
    }
    
    return tensor;
}

// InferenceWorker实现
InferenceWorker::InferenceWorker(NLKiller* killer, std::vector<ImageInfo>* images)
    : killer(killer), images(images), shouldStop(false) {
}

InferenceWorker::~InferenceWorker() {
    shouldStop = true;
    
    // 等待所有解码线程结束
    for (auto& thread : decoderThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    decoderThreads.clear();
}

YTensor<unsigned char, 3> InferenceWorker::loadImageToTensor(const QString& path) {
    // 禁用Qt的ICC警告
    qputenv("QT_LOGGING_RULES", "qt.gui.icc.debug=false");
    
    QImage qimg(path);
    if (qimg.isNull()) {
        return YTensor<unsigned char, 3>();
    }
    
    return qImageToYTensor(qimg);
}

void InferenceWorker::decoderWorker(int startIndex, int endIndex) {
    for (int i = startIndex; i < endIndex && i < static_cast<int>(images->size()) && !shouldStop; ++i) {
        // 加载图片
        auto tensor = std::make_shared<YTensor<unsigned char, 3>>(loadImageToTensor((*images)[i].filePath));
        
        if (tensor && tensor->data != nullptr) {
            // 使用lock_guard锁住队列，添加解码结果
            std::lock_guard<std::mutex> lock(queueMutex);
            decodedQueue.emplace(static_cast<int>(i), tensor);
        }
    }
}

void InferenceWorker::inferenceLoop() {
    auto startTime = std::chrono::high_resolution_clock::now();
    int processedCount = 0;
    int totalCount = static_cast<int>(images->size());

    while (processedCount < totalCount && !shouldStop) {
        std::vector<DecodedResult> batch;
        
        // 从队列中取出所有已解码的图片
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!decodedQueue.empty()) {
                batch.push_back(decodedQueue.front());
                decodedQueue.pop();
            }
        }
        
        // 对每个解码好的图片进行推理
        for (const auto& decoded : batch) {
            if (shouldStop) break;
            
            float confidence = killer->infer(*decoded.tensor);
            processedCount++;
            
            // 发出单个图片处理完成的信号
            Q_EMIT imageProcessed(decoded.index, confidence);
            
            // 计算并发出进度更新信号
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(currentTime - startTime).count();
            float speed = elapsed > 0 ? processedCount / elapsed : 0.0f;
            
            Q_EMIT progressUpdated(processedCount, totalCount, speed);
        }
        
        // 如果没有可处理的图片，短暂等待
        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    Q_EMIT allImagesProcessed();
}

void InferenceWorker::processImages() {
    if (!images || images->empty()) {
        Q_EMIT allImagesProcessed();
        return;
    }
    
    shouldStop = false;
    
    // 清空解码队列
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!decodedQueue.empty()) {
            decodedQueue.pop();
        }
    }
    
    // 计算解码线程数量（逻辑核心数-2，最少1个）
    int numDecoderThreads = std::max(1, (int)std::thread::hardware_concurrency() - 2);
    int imagesPerThread = (images->size() + numDecoderThreads - 1) / numDecoderThreads;
    
    // 启动解码线程，每个线程处理一段图片
    decoderThreads.clear();
    for (int i = 0; i < numDecoderThreads; ++i) {
        int startIndex = i * imagesPerThread;
        int endIndex = std::min(startIndex + imagesPerThread, (int)images->size());
        
        if (startIndex < static_cast<int>(images->size())) {
            decoderThreads.emplace_back(&InferenceWorker::decoderWorker, this, startIndex, endIndex);
        }
    }
    
    // 在当前线程中运行推理循环
    inferenceLoop();
}

void InferenceWorker::processSingleImage(int index) {
    if (index < 0 || index >= static_cast<int>(images->size())) {
        return;
    }
    
    auto tensor = loadImageToTensor((*images)[index].filePath);
    if (tensor.data == nullptr) {
        return;
    }
    
    float result = killer->infer(tensor);
    Q_EMIT singleImageProcessed(index, result);
}

// NLKillerGUI实现
NLKillerGUI::NLKillerGUI(QWidget *parent)
    : QMainWindow(parent), currentImageIndex(0), confidenceThreshold(0.5f), 
      workerThread(nullptr), worker(nullptr) {
    
    // 初始化支持的图像格式
    supportedFormats << "jpg" << "jpeg" << "png" << "bmp" << "tiff" << "tga";
    
    // 初始化模型路径
    modelPaths << "../models/helicopter_simplified.onnx" 
               << "../models/NLK-s_simplified.onnx" 
               << "../models/yvgg_simplified.onnx";
    
    // 初始化AI推理器
    killer = std::make_unique<NLKiller>(false);
    
    // 设置UI
    setupUI();
    
    // 默认加载第一个模型
    loadModel(modelPaths[0]);
    
    // 设置窗口属性
    setWindowTitle("NLKiller GUI");
    setMinimumSize(1200, 800);
    resize(1400, 900);
    
    // 设置焦点策略以接收键盘事件
    setFocusPolicy(Qt::StrongFocus);
}

NLKillerGUI::~NLKillerGUI() {
    // 安全地停止和清理工作线程
    if (workerThread != nullptr) {
        if (workerThread->isRunning()) {
            workerThread->quit();
            workerThread->wait(3000); // 最多等待3秒
        }
        delete workerThread;
        workerThread = nullptr;
    }
    // worker已经被信号机制删除了
}

void NLKillerGUI::setupUI() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    mainLayout = new QHBoxLayout(centralWidget);
    
    setupImagePreview();
    setupControlPanel();
}

void NLKillerGUI::setupImagePreview() {
    // 左侧图像预览区域
    imageScrollArea = new QScrollArea();
    imageLabel = new QLabel();
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setMinimumSize(600, 600);
    imageLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    imageLabel->setText("请选择文件夹加载图像");
    
    imageScrollArea->setWidget(imageLabel);
    imageScrollArea->setWidgetResizable(true);
    
    mainLayout->addWidget(imageScrollArea, 2);
}

void NLKillerGUI::setupControlPanel() {
    // 右侧控制面板
    controlPanel = new QWidget();
    controlPanel->setMaximumWidth(350);
    controlLayout = new QVBoxLayout(controlPanel);
    
    // 打开文件夹按钮
    openFolderBtn = new QPushButton("打开文件夹");
    openFolderBtn->setFocusPolicy(Qt::NoFocus);
    connect(openFolderBtn, &QPushButton::clicked, this, &NLKillerGUI::openFolder);
    controlLayout->addWidget(openFolderBtn);
    
    // 一键查杀按钮
    batchInferenceBtn = new QPushButton("一键查杀");
    batchInferenceBtn->setEnabled(false);
    batchInferenceBtn->setFocusPolicy(Qt::NoFocus);
    connect(batchInferenceBtn, &QPushButton::clicked, this, &NLKillerGUI::batchInference);
    controlLayout->addWidget(batchInferenceBtn);
    
    // 进度条和速度显示
    progressWidget = new QWidget();
    progressLayout = new QHBoxLayout(progressWidget);
    progressLayout->setContentsMargins(0, 0, 0, 0);
    
    progressBar = new QProgressBar();
    progressBar->setVisible(false);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    
    speedLabel = new QLabel("0.0 img/s");
    speedLabel->setVisible(false);
    speedLabel->setMinimumWidth(80);
    speedLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    
    progressLayout->addWidget(progressBar);
    progressLayout->addWidget(speedLabel);
    controlLayout->addWidget(progressWidget);
    
    // 置信度阈值滑条
    confidenceLabel = new QLabel("置信度阈值:");
    controlLayout->addWidget(confidenceLabel);
    
    QHBoxLayout* sliderLayout = new QHBoxLayout();
    confidenceSlider = new QSlider(Qt::Horizontal);
    confidenceSlider->setRange(0, 1000);
    confidenceSlider->setValue(500);
    confidenceSlider->setFocusPolicy(Qt::NoFocus);
    connect(confidenceSlider, &QSlider::valueChanged, this, &NLKillerGUI::onConfidenceChanged);
    
    confidenceValueLabel = new QLabel("0.500");
    confidenceValueLabel->setMinimumWidth(50);
    
    sliderLayout->addWidget(confidenceSlider);
    sliderLayout->addWidget(confidenceValueLabel);
    controlLayout->addLayout(sliderLayout);
    
    // 刷新按钮
    refreshBtn = new QPushButton("刷新");
    refreshBtn->setEnabled(false);
    refreshBtn->setFocusPolicy(Qt::NoFocus);
    connect(refreshBtn, &QPushButton::clicked, this, &NLKillerGUI::refreshResults);
    controlLayout->addWidget(refreshBtn);
    
    // 图像列表表格
    imageTable = new QTableWidget();
    imageTable->setColumnCount(2);
    imageTable->setHorizontalHeaderLabels(QStringList() << "文件名" << "状态");
    imageTable->horizontalHeader()->setStretchLastSection(false);
    imageTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    imageTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
    imageTable->setColumnWidth(1, 60);
    imageTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    imageTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    imageTable->setFocusPolicy(Qt::NoFocus);
    connect(imageTable, &QTableWidget::itemSelectionChanged, this, &NLKillerGUI::onTableSelectionChanged);
    controlLayout->addWidget(imageTable);
    
    // 当前图像状态标签
    currentImageStatusLabel = new QLabel("请加载图像");
    currentImageStatusLabel->setAlignment(Qt::AlignCenter);
    currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; }");
    controlLayout->addWidget(currentImageStatusLabel);
    
    // 统计信息标签
    statisticsLabel = new QLabel("统计: 未加载图像");
    statisticsLabel->setAlignment(Qt::AlignCenter);
    statisticsLabel->setStyleSheet("QLabel { font-size: 12px; color: #666;}");
    controlLayout->addWidget(statisticsLabel);
    
    // 导出按钮
    exportBtn = new QPushButton("导出结果");
    exportBtn->setEnabled(false);
    exportBtn->setFocusPolicy(Qt::NoFocus);
    connect(exportBtn, &QPushButton::clicked, this, &NLKillerGUI::exportResults);
    controlLayout->addWidget(exportBtn);
    
    // 模型选择
    modelLabel = new QLabel("模型选择:");
    controlLayout->addWidget(modelLabel);
    
    modelComboBox = new QComboBox();
    modelComboBox->addItems(QStringList() << "质量最高 (helicopter)" << "平衡 (NLK-s)" << "速度最快 (yvgg)");
    modelComboBox->setFocusPolicy(Qt::NoFocus);
    connect(modelComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &NLKillerGUI::onModelChanged);
    controlLayout->addWidget(modelComboBox);
    
    mainLayout->addWidget(controlPanel, 1);
}

void NLKillerGUI::openFolder() {
    QString folderPath = QFileDialog::getExistingDirectory(this, "选择图像文件夹");
    if (!folderPath.isEmpty()) {
        loadImagesFromFolder(folderPath);
    }
}

void NLKillerGUI::loadImagesFromFolder(const QString& folderPath) {
    images.clear();
    currentImageIndex = 0;
    
    QDir dir(folderPath);
    QStringList filters;
    for (const QString& format : supportedFormats) {
        filters << QString("*.%1").arg(format);
        filters << QString("*.%1").arg(format.toUpper());
    }
    
    QStringList fileNames = dir.entryList(filters, QDir::Files, QDir::Name);
    
    for (const QString& fileName : fileNames) {
        ImageInfo info;
        info.fileName = fileName;
        info.filePath = dir.absoluteFilePath(fileName);
        images.push_back(info);
    }
    
    if (!images.empty()) {
        batchInferenceBtn->setEnabled(true);
        refreshBtn->setEnabled(true);
        exportBtn->setEnabled(true);
        updateImageTable();
        updateImagePreview();
        updateCurrentImageStatus();
        updateStatistics();
        
        // 对第一张图片立即进行推理
        if (!images.empty()) {
            processCurrentImage();
        }
    } else {
        QMessageBox::information(this, "提示", "未找到支持的图像文件");
        updateStatistics();
    }
}

void NLKillerGUI::updateImagePreview() {
    if (images.empty() || currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        imageLabel->setText("无图像");
        return;
    }
    
    QPixmap pixmap = loadImageAsPixmap(images[currentImageIndex].filePath);
    if (!pixmap.isNull()) {
        // 缩放图像以适应预览区域，保持宽高比
        QPixmap scaledPixmap = pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        imageLabel->setPixmap(scaledPixmap);
    } else {
        imageLabel->setText("无法加载图像");
    }
    
    // 高亮当前行
    imageTable->selectRow(currentImageIndex);
}

void NLKillerGUI::updateImageTable() {
    imageTable->setRowCount(static_cast<int>(images.size()));

    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        QTableWidgetItem* nameItem = new QTableWidgetItem(images[i].fileName);
        imageTable->setItem(i, 0, nameItem);
        
        QString statusText = "?";
        if (images[i].processed) {
            statusText = images[i].isPositive ? "✅" : "❌";
        }
        QTableWidgetItem* statusItem = new QTableWidgetItem(statusText);
        statusItem->setTextAlignment(Qt::AlignCenter);
        imageTable->setItem(i, 1, statusItem);
    }
}

void NLKillerGUI::updateCurrentImageStatus() {
    if (images.empty() || currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        currentImageStatusLabel->setText("请加载图像");
        currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: black; }");
        return;
    }
    
    const ImageInfo& info = images[currentImageIndex];
    if (!info.processed) {
        currentImageStatusLabel->setText("未处理");
        currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: gray; }");
    } else if (info.isPositive) {
        currentImageStatusLabel->setText("NAILONG FOUND");
        currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: red; }");
    } else {
        currentImageStatusLabel->setText("pass");
        currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: green; }");
    }
}

void NLKillerGUI::switchToImage(int index) {
    if (index < 0 || index >= static_cast<int>(images.size())) {
        return;
    }
    
    currentImageIndex = index;
    updateImagePreview();
    updateCurrentImageStatus();
    
    // 如果当前图像未处理过，进行单张推理
    if (!images[currentImageIndex].processed) {
        processCurrentImage();
    }
}

void NLKillerGUI::processCurrentImage() {
    if (currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        return;
    }
    
    if (images[currentImageIndex].processed) {
        return; // 已经处理过了
    }
    
    // 直接同步推理，避免快速切换时的漏识别问题
    auto tensor = InferenceWorker::loadImageToTensor(images[currentImageIndex].filePath);
    if (tensor.data != nullptr) {
        float confidence = killer->infer(tensor);
        onSingleImageProcessed(currentImageIndex, confidence);
    }
}

void NLKillerGUI::batchInference() {
    if (images.empty()) {
        return;
    }
    
    batchInferenceBtn->setEnabled(false);
    batchInferenceBtn->setText("推理中...");
    
    // 显示进度条
    progressBar->setValue(0);
    progressBar->setVisible(true);
    speedLabel->setText("0.0 img/s");
    speedLabel->setVisible(true);
    
    // 停止之前的推理线程
    if (workerThread != nullptr) {
        if (workerThread->isRunning()) {
            workerThread->quit();
            workerThread->wait(3000); // 最多等待3秒
        }
        delete workerThread;
        workerThread = nullptr;
    }
    
    // 创建新的线程和worker
    workerThread = new QThread(this);
    worker = new InferenceWorker(killer.get(), &images);
    worker->moveToThread(workerThread);
    
    connect(workerThread, &QThread::started, worker, &InferenceWorker::processImages);
    connect(worker, &InferenceWorker::imageProcessed, this, &NLKillerGUI::onImageProcessed);
    connect(worker, &InferenceWorker::progressUpdated, this, &NLKillerGUI::onProgressUpdated);
    connect(worker, &InferenceWorker::allImagesProcessed, this, &NLKillerGUI::onAllImagesProcessed);
    
    // 确保worker在线程结束时被清理
    connect(workerThread, &QThread::finished, worker, &QObject::deleteLater);
    
    workerThread->start();
}

void NLKillerGUI::refreshResults() {
    for (ImageInfo& info : images) {
        if (info.processed) {
            info.isPositive = (info.confidence >= confidenceThreshold);
        }
    }
    updateImageTable();
    updateCurrentImageStatus();
    updateStatistics();
}

void NLKillerGUI::onConfidenceChanged(int value) {
    confidenceThreshold = value / 1000.0f;
    QString text = QString::number(confidenceThreshold, 'f', 3);
    confidenceValueLabel->setText(text);
}

void NLKillerGUI::onImageProcessed(int index, float confidence) {
    if (index >= 0 && index < static_cast<int>(images.size())) {
        images[index].confidence = confidence;
        images[index].processed = true;
        images[index].isPositive = (confidence >= confidenceThreshold);
        
        // 更新表格中的单个项目
        QString statusText = images[index].isPositive ? "✅" : "❌";
        QTableWidgetItem* statusItem = new QTableWidgetItem(statusText);
        statusItem->setTextAlignment(Qt::AlignCenter);
        imageTable->setItem(index, 1, statusItem);
        
        // 如果是当前显示的图像，更新状态
        if (index == currentImageIndex) {
            updateCurrentImageStatus();
        }
        
        updateStatistics();
    }
}

void NLKillerGUI::onAllImagesProcessed() {
    batchInferenceBtn->setEnabled(true);
    batchInferenceBtn->setText("一键查杀");
    
    // 隐藏进度条
    progressBar->setVisible(false);
    speedLabel->setVisible(false);
    
    updateStatistics();
    QMessageBox::information(this, "完成", "批量推理完成！");
}

void NLKillerGUI::onSingleImageProcessed(int index, float confidence) {
    if (index >= 0 && index < static_cast<int>(images.size())) {
        images[index].confidence = confidence;
        images[index].processed = true;
        images[index].isPositive = (confidence >= confidenceThreshold);
        
        updateImageTable();
        updateCurrentImageStatus();
        updateStatistics();
    }
}

void NLKillerGUI::onModelChanged(int index) {
    if (index >= 0 && index < modelPaths.size()) {
        loadModel(modelPaths[index]);
        
        // 清空所有图片的推理结果
        for (ImageInfo& info : images) {
            info.processed = false;
            info.confidence = 0.0f;
            info.isPositive = false;
        }
        
        // 更新表格显示
        updateImageTable();
        
        // 如果有当前图片，立即重新推理
        if (!images.empty() && currentImageIndex >= 0 && currentImageIndex < static_cast<int>(images.size())) {
            processCurrentImage();
        }
    }
}

void NLKillerGUI::loadModel(const QString& modelPath) {
    if (killer->loadModel(modelPath.toStdString())) {
        // 模型加载成功，可以在状态栏或其他地方显示提示
        setWindowTitle(QString("NLKiller GUI - %1").arg(QFileInfo(modelPath).baseName()));
    } else {
        QMessageBox::critical(this, "错误", QString("无法加载模型: %1").arg(modelPath));
    }
}

void NLKillerGUI::exportResults() {
    if (images.empty()) {
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "导出结果", 
                                                   QDir::homePath() + "/nlkiller_results.csv",
                                                   "CSV files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "错误", "无法创建文件");
        return;
    }
    
    QTextStream out(&file);
    out << "文件路径,置信度,判定结果\n";
    
    for (const ImageInfo& info : images) {
        if (info.processed) {
            out << info.filePath << "," 
                << QString::number(info.confidence, 'f', 6) << ","
                << (info.isPositive ? "True" : "False") << "\n";
        }
    }
    
    QMessageBox::information(this, "完成", "结果已导出到: " + fileName);
}

QPixmap NLKillerGUI::loadImageAsPixmap(const QString& path) {
    // 禁用Qt的ICC警告
    qputenv("QT_LOGGING_RULES", "qt.gui.icc.debug=false");
    
    QPixmap pixmap;
    if (!pixmap.load(path)) {
        // 如果直接加载失败，尝试先加载为QImage再转换
        QImage qimg(path);
        if (!qimg.isNull()) {
            pixmap = QPixmap::fromImage(qimg);
        }
    }
    return pixmap;
}

void NLKillerGUI::keyPressEvent(QKeyEvent *event) {
    if (images.empty()) {
        QMainWindow::keyPressEvent(event);
        return;
    }
    
    switch (event->key()) {
        case Qt::Key_S:
        case Qt::Key_D:
        case Qt::Key_Down:
        case Qt::Key_Right:
            switchToImage((currentImageIndex + 1) % images.size());
            break;
        case Qt::Key_W:
        case Qt::Key_A:
        case Qt::Key_Up:
        case Qt::Key_Left:
            switchToImage((currentImageIndex - 1 + images.size()) % images.size());
            break;
        default:
            QMainWindow::keyPressEvent(event);
    }
}

void NLKillerGUI::onProgressUpdated(int current, int total, float speed) {
    if (total > 0) {
        int percentage = (current * 100) / total;
        progressBar->setValue(percentage);
    }
    
    QString speedText = QString::number(speed, 'f', 1) + " img/s";
    speedLabel->setText(speedText);
}

void NLKillerGUI::onTableSelectionChanged() {
    QList<QTableWidgetItem*> selectedItems = imageTable->selectedItems();
    if (!selectedItems.isEmpty()) {
        int row = selectedItems.first()->row();
        if (row >= 0 && row < static_cast<int>(images.size()) && row != currentImageIndex) {
            switchToImage(row);
        }
    }
}

void NLKillerGUI::updateStatistics() {
    if (images.empty()) {
        statisticsLabel->setText("统计: 未加载图像");
        return;
    }
    
    int unscanned = 0;
    int passed = 0;
    int detected = 0;
    
    for (const auto& img : images) {
        if (!img.processed) {
            unscanned++;
        } else if (img.isPositive) {
            detected++;
        } else {
            passed++;
        }
    }
    QString statsText = QString("未扫描 %1 | 通过 %2 | 发现奶龙 %3").arg(unscanned).arg(passed).arg(detected);

    statisticsLabel->setText(statsText);
}
