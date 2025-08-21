// ���Qt��TBB��emit��ͻ
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

// ��QImageת��ΪYTensor<u_char, 3>
YTensor<u_char, 3> qImageToYTensor(const QImage& qimg) {
    // ȷ��ͼ����RGB��ʽ
    QImage rgbImage = qimg.convertToFormat(QImage::Format_RGB888);
    
    int height = rgbImage.height();
    int width = rgbImage.width();
    int channels = 3;
    
    YTensor<u_char, 3> tensor(height, width, channels);
    
    // ������������
    const uchar* srcData = rgbImage.constBits();
    int bytesPerLine = rgbImage.bytesPerLine();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uchar* pixel = srcData + y * bytesPerLine + x * channels;
            // QImage��RGB˳���������Ҫ��һ��
            tensor.at(y, x, 0) = pixel[0]; // R
            tensor.at(y, x, 1) = pixel[1]; // G  
            tensor.at(y, x, 2) = pixel[2]; // B
        }
    }
    
    return tensor;
}

// InferenceWorkerʵ��
InferenceWorker::InferenceWorker(NLKiller* killer, std::vector<ImageInfo>* images)
    : killer(killer), images(images), shouldStop(false) {
}

InferenceWorker::~InferenceWorker() {
    shouldStop = true;
    
    // �ȴ����н����߳̽���
    for (auto& thread : decoderThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    decoderThreads.clear();
}

YTensor<unsigned char, 3> InferenceWorker::loadImageToTensor(const QString& path) {
    // ����Qt��ICC����
    qputenv("QT_LOGGING_RULES", "qt.gui.icc.debug=false");
    
    QImage qimg(path);
    if (qimg.isNull()) {
        return YTensor<unsigned char, 3>(1,1,1);
    }
    
    return qImageToYTensor(qimg);
}

void InferenceWorker::decoderWorker(int startIndex, int endIndex) {
    for (int i = startIndex; i < endIndex && i < static_cast<int>(images->size()) && !shouldStop; ++i) {
        // ����ͼƬ
        auto tensor = std::make_shared<YTensor<unsigned char, 3>>(loadImageToTensor((*images)[i].filePath));
        
        if (tensor && tensor->data != nullptr) {
            // ����,��ͣ���
            while (pause) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // ʹ��lock_guard��ס���У���ӽ�����
            std::lock_guard<std::mutex> lock(queueMutex);
            decodedQueue.emplace(static_cast<int>(i), tensor);
            pause = static_cast<int>(decodedQueue.size()) >= buffer_size;
        }
    }
}

void InferenceWorker::inferenceLoop() {
    auto startTime = std::chrono::high_resolution_clock::now();
    int processedCount = 0;
    int totalCount = static_cast<int>(images->size());

    while (processedCount < totalCount && !shouldStop) {
        std::vector<DecodedResult> batch;
        
        // �Ӷ�����ȡ�������ѽ����ͼƬ
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!decodedQueue.empty()) {
                batch.push_back(decodedQueue.front());
                decodedQueue.pop();
            }
            pause = false;
        }
        
        // ��ÿ������õ�ͼƬ��������
        // for (int a = 0; a < batch.size(); a++) {
        //     const auto& decoded = batch[a];
        //     if (shouldStop) break;
            
        //     float confidence = killer->infer(*decoded.tensor);
        //     processedCount++;
            
        //     // ��������ͼƬ������ɵ��ź�
        //     Q_EMIT imageProcessed(decoded.index, confidence);
            
        //     // ���㲢�������ȸ����ź�
        //     auto currentTime = std::chrono::high_resolution_clock::now();
        //     auto elapsed = std::chrono::duration<float>(currentTime - startTime).count();
        //     float speed = elapsed > 0 ? processedCount / elapsed : 0.0f;
            
        //     Q_EMIT progressUpdated(processedCount, totalCount, speed);
        // }
        if(!shouldStop){
            std::vector<YTensor<u_char, 3>> tensors;
            tensors.reserve(batch.size());
            for (int a = 0; a < batch.size(); a++) {
                tensors.emplace_back((*batch[a].tensor).move());
            }
            auto confs = killer->infer(tensors, NLKiller::InferenceMode::ASYNC_MULTI);
            processedCount += confs.size();
			for (int a = 0; a < confs.size(); a++) {
				// ��������ͼƬ������ɵ��ź�
				Q_EMIT imageProcessed(batch[a].index, confs[a]);
			}
            // ���㲢�������ȸ����ź�
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(currentTime - startTime).count();
            float speed = elapsed > 0 ? processedCount / elapsed : 0.0f;

            Q_EMIT progressUpdated(processedCount, totalCount, speed);
        }
        
        
        // ���û�пɴ����ͼƬ�����ݵȴ�
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
    
    // ��ս������
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!decodedQueue.empty()) {
            decodedQueue.pop();
        }
    }
    
    
    // ��������߳��������߼�������-2������1����
    int numDecoderThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2 - 1);
    int imagesPerThread = (images->size() + numDecoderThreads - 1) / numDecoderThreads;
    
    // ���������̣߳�ÿ���̴߳���һ��ͼƬ
    decoderThreads.clear();
    for (int i = 0; i < numDecoderThreads; ++i) {
        int startIndex = i * imagesPerThread;
        int endIndex = std::min(startIndex + imagesPerThread, (int)images->size());
        
        if (startIndex < static_cast<int>(images->size())) {
            decoderThreads.emplace_back(&InferenceWorker::decoderWorker, this, startIndex, endIndex);
        }
    }
    
    // �ڵ�ǰ�߳�����������ѭ��
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

// NLKillerGUIʵ��
NLKillerGUI::NLKillerGUI(QWidget *parent)
    : QMainWindow(parent), currentImageIndex(0), confidenceThreshold(0.5f), 
      workerThread(nullptr), worker(nullptr) {
    
    // ��ʼ��֧�ֵ�ͼ���ʽ
    supportedFormats << "jpg" << "jpeg" << "png" << "bmp" << "tiff" << "tga";
    
    // ��ʼ��ģ��·��
    modelPaths << "../models/helicopter_simplified.onnx" 
               << "../models/NLK-s_simplified.onnx" 
               << "../models/yvgg_simplified.onnx";
    
    // ��ʼ��AI������
    killer = std::make_unique<NLKiller>(false);
    killer->setNumThreads(4);
    // killer->setDevice(NLKiller::DeviceType::GPU, 1);
    
    // ����UI
    setupUI();
    
    // Ĭ�ϼ��ص�һ��ģ��
    loadModel(modelPaths[0]);
    
    // ���ô�������
    setWindowTitle("NLKiller GUI");
    setMinimumSize(1200, 800);
    resize(1400, 900);
    
    // ���ý�������Խ��ռ����¼�
    setFocusPolicy(Qt::StrongFocus);
}

NLKillerGUI::~NLKillerGUI() {
    // ��ȫ��ֹͣ���������߳�
    if (workerThread != nullptr) {
        if (workerThread->isRunning()) {
            workerThread->quit();
            workerThread->wait(3000); // ���ȴ�3��
        }
        delete workerThread;
        workerThread = nullptr;
    }
    // worker�Ѿ����źŻ���ɾ����
}

void NLKillerGUI::setupUI() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    mainLayout = new QHBoxLayout(centralWidget);
    
    setupImagePreview();
    setupControlPanel();
}

void NLKillerGUI::setupImagePreview() {
    // ���ͼ��Ԥ������
    imageScrollArea = new QScrollArea();
    imageLabel = new QLabel();
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setMinimumSize(600, 600);
    imageLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    imageLabel->setText("\350\257\267\351\200\211\346\213\251\346\226\207\344\273\266\345\244\271\345\212\240\350\275\275\345\233\276\345\203\217");
    
    imageScrollArea->setWidget(imageLabel);
    imageScrollArea->setWidgetResizable(true);
    
    mainLayout->addWidget(imageScrollArea, 2);
}

void NLKillerGUI::setupControlPanel() {
    // �Ҳ�������
    controlPanel = new QWidget();
    controlPanel->setMaximumWidth(350);
    controlLayout = new QVBoxLayout(controlPanel);
    
    // ���ļ��а�ť
    openFolderBtn = new QPushButton("\346\211\223\345\274\200\346\226\207\344\273\266\345\244\271");
    openFolderBtn->setFocusPolicy(Qt::NoFocus);
    connect(openFolderBtn, &QPushButton::clicked, this, &NLKillerGUI::openFolder);
    controlLayout->addWidget(openFolderBtn);
    
    // һ����ɱ��ť
    batchInferenceBtn = new QPushButton("\344\270\200\351\224\256\346\237\245\346\235\200");
    batchInferenceBtn->setEnabled(false);
    batchInferenceBtn->setFocusPolicy(Qt::NoFocus);
    connect(batchInferenceBtn, &QPushButton::clicked, this, &NLKillerGUI::batchInference);
    controlLayout->addWidget(batchInferenceBtn);
    
    // ���������ٶ���ʾ
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
    
    // ���Ŷ���ֵ����
    confidenceLabel = new QLabel("\347\275\256\344\277\241\345\272\246\351\230\210\345\200\274\72");
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
    
    // ˢ�°�ť
    refreshBtn = new QPushButton("\345\210\267\346\226\260");
    refreshBtn->setEnabled(false);
    refreshBtn->setFocusPolicy(Qt::NoFocus);
    connect(refreshBtn, &QPushButton::clicked, this, &NLKillerGUI::refreshResults);
    controlLayout->addWidget(refreshBtn);
    
    // ͼ���б���
    imageTable = new QTableWidget();
    imageTable->setColumnCount(2);
    imageTable->setHorizontalHeaderLabels(QStringList() << "\346\226\207\344\273\266\345\220\215" << "\347\212\266\346\200\201");
    imageTable->horizontalHeader()->setStretchLastSection(false);
    imageTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    imageTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
    imageTable->setColumnWidth(1, 60);
    imageTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    imageTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    imageTable->setFocusPolicy(Qt::NoFocus);
    connect(imageTable, &QTableWidget::itemSelectionChanged, this, &NLKillerGUI::onTableSelectionChanged);
    controlLayout->addWidget(imageTable);
    
    // ��ǰͼ��״̬��ǩ
    currentImageStatusLabel = new QLabel("\350\257\267\345\212\240\350\275\275\345\233\276\345\203\217");
    currentImageStatusLabel->setAlignment(Qt::AlignCenter);
    currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; }");
    controlLayout->addWidget(currentImageStatusLabel);
    
    // ͳ����Ϣ��ǩ
    statisticsLabel = new QLabel("\347\273\237\350\256\241\72\40\346\234\252\345\212\240\350\275\275\345\233\276\345\203\217");
    statisticsLabel->setAlignment(Qt::AlignCenter);
    statisticsLabel->setStyleSheet("QLabel { font-size: 12px; color: #666;}");
    controlLayout->addWidget(statisticsLabel);
    
    // ������ť
    exportBtn = new QPushButton("\345\257\274\345\207\272\347\273\223\346\236\234");
    exportBtn->setEnabled(false);
    exportBtn->setFocusPolicy(Qt::NoFocus);
    connect(exportBtn, &QPushButton::clicked, this, &NLKillerGUI::exportResults);
    controlLayout->addWidget(exportBtn);
    
    // ģ��ѡ��
    modelLabel = new QLabel("\346\250\241\345\236\213\351\200\211\346\213\251\72");
    controlLayout->addWidget(modelLabel);
    
    modelComboBox = new QComboBox();
    modelComboBox->addItems(QStringList() << "\350\264\250\351\207\217\346\234\200\351\253\230\40\50\150\145\154\151\143\157\160\164\145\162\51"
        << "\345\271\263\350\241\241\40\50\116\114\113\55\163\51" << "\351\200\237\345\272\246\346\234\200\345\277\253\40\50\171\166\147\147\51");
    modelComboBox->setFocusPolicy(Qt::NoFocus);
    connect(modelComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &NLKillerGUI::onModelChanged);
    controlLayout->addWidget(modelComboBox);
    
    mainLayout->addWidget(controlPanel, 1);
}

void NLKillerGUI::openFolder() {
    QString folderPath = QFileDialog::getExistingDirectory(this, "\351\200\211\346\213\251\345\233\276\345\203\217\346\226\207\344\273\266\345\244\271");
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
        
        // �Ե�һ��ͼƬ������������
        if (!images.empty()) {
            processCurrentImage();
        }
    } else {
        QMessageBox::information(this, "\346\217\220\347\244\272", "\346\234\252\346\211\276\345\210\260\346\224\257\346\214\201\347\232\204\345\233\276\345\203\217\346\226\207\344\273\266");
        updateStatistics();
    }
}

void NLKillerGUI::updateImagePreview() {
    if (images.empty() || currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        imageLabel->setText("\346\227\240\345\233\276\345\203\217");
        return;
    }
    
    QPixmap pixmap = loadImageAsPixmap(images[currentImageIndex].filePath);
    if (!pixmap.isNull()) {
        // ����ͼ������ӦԤ�����򣬱��ֿ�߱�
        QPixmap scaledPixmap = pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        imageLabel->setPixmap(scaledPixmap);
    } else {
        imageLabel->setText("\346\227\240\346\263\225\345\212\240\350\275\275\345\233\276\345\203\217");
    }
    
    // ������ǰ��
    imageTable->selectRow(currentImageIndex);
}

void NLKillerGUI::updateImageTable() {
    imageTable->setRowCount(static_cast<int>(images.size()));

    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        QTableWidgetItem* nameItem = new QTableWidgetItem(images[i].fileName);
        imageTable->setItem(i, 0, nameItem);
        
        QString statusText = "?";
        if (images[i].processed) {
            statusText = images[i].isPositive ? "\342\234\205" : "\342\235\214";
        }
        QTableWidgetItem* statusItem = new QTableWidgetItem(statusText);
        statusItem->setTextAlignment(Qt::AlignCenter);
        imageTable->setItem(i, 1, statusItem);
    }
}

void NLKillerGUI::updateCurrentImageStatus() {
    if (images.empty() || currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        currentImageStatusLabel->setText("\350\257\267\345\212\240\350\275\275\345\233\276\345\203\217");
        currentImageStatusLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: black; }");
        return;
    }
    
    const ImageInfo& info = images[currentImageIndex];
    if (!info.processed) {
        currentImageStatusLabel->setText("\346\234\252\345\244\204\347\220\206");
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
    
    // �����ǰͼ��δ����������е�������
    if (!images[currentImageIndex].processed) {
        processCurrentImage();
    }
}

void NLKillerGUI::processCurrentImage() {
    if (currentImageIndex < 0 || currentImageIndex >= static_cast<int>(images.size())) {
        return;
    }
    
    if (images[currentImageIndex].processed) {
        return; // �Ѿ��������
    }
    
    // ֱ��ͬ��������������л�ʱ��©ʶ������
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
    batchInferenceBtn->setText("\346\216\250\347\220\206\344\270\255\56\56\56");
    modelComboBox->setEnabled(false);
    
    // ��ʾ������
    progressBar->setValue(0);
    progressBar->setVisible(true);
    speedLabel->setText("0.0 img/s");
    speedLabel->setVisible(true);
    
    // ֹ֮ͣǰ�������߳�
    if (workerThread != nullptr) {
        if (workerThread->isRunning()) {
            workerThread->quit();
            workerThread->wait(3000); // ���ȴ�3��
        }
        delete workerThread;
        workerThread = nullptr;
    }
    
    // �����µ��̺߳�worker
    workerThread = new QThread(this);
    worker = new InferenceWorker(killer.get(), &images);
    worker->moveToThread(workerThread);
    
    connect(workerThread, &QThread::started, worker, &InferenceWorker::processImages);
    connect(worker, &InferenceWorker::imageProcessed, this, &NLKillerGUI::onImageProcessed);
    connect(worker, &InferenceWorker::progressUpdated, this, &NLKillerGUI::onProgressUpdated);
    connect(worker, &InferenceWorker::allImagesProcessed, this, &NLKillerGUI::onAllImagesProcessed);
    
    // ȷ��worker���߳̽���ʱ������
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
        
        // ���±���еĵ�����Ŀ
        QString statusText = images[index].isPositive ? "\342\234\205" : "\342\235\214";
        QTableWidgetItem* statusItem = new QTableWidgetItem(statusText);
        statusItem->setTextAlignment(Qt::AlignCenter);
        imageTable->setItem(index, 1, statusItem);
        
        // ����ǵ�ǰ��ʾ��ͼ�񣬸���״̬
        if (index == currentImageIndex) {
            updateCurrentImageStatus();
        }
        
        updateStatistics();
    }
}

void NLKillerGUI::onAllImagesProcessed() {
    batchInferenceBtn->setEnabled(true);
    batchInferenceBtn->setText("\344\270\200\351\224\256\346\237\245\346\235\200");
    
    // ���ؽ�����
    progressBar->setVisible(false);
    speedLabel->setVisible(false);
    modelComboBox->setEnabled(true);
    
    updateStatistics();
    QMessageBox::information(this, "\345\256\214\346\210\220", "\346\211\271\351\207\217\346\216\250\347\220\206\345\256\214\346\210\220\357\274\201");
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
        
        // �������ͼƬ��������
        for (ImageInfo& info : images) {
            info.processed = false;
            info.confidence = 0.0f;
            info.isPositive = false;
        }
        
        // ���±����ʾ
        updateImageTable();
        
        // ����е�ǰͼƬ��������������
        if (!images.empty() && currentImageIndex >= 0 && currentImageIndex < static_cast<int>(images.size())) {
            processCurrentImage();
        }
    }
}

void NLKillerGUI::loadModel(const QString& modelPath) {
    if (killer->loadModel(modelPath.toStdString())) {
        // ģ�ͼ��سɹ���������״̬���������ط���ʾ��ʾ
        setWindowTitle(QString("NLKiller GUI - %1").arg(QFileInfo(modelPath).baseName()));
    } else {
        QMessageBox::critical(this, "\351\224\231\350\257\257", QString("\346\227\240\346\263\225\345\212\240\350\275\275\346\250\241\345\236\213\72\40%1").arg(modelPath));
    }
}

void NLKillerGUI::exportResults() {
    if (images.empty()) {
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "\345\257\274\345\207\272\347\273\223\346\236\234", 
                                                   QDir::homePath() + "/nlkiller_results.csv",
                                                   "CSV files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "\351\224\231\350\257\257", "\346\227\240\346\263\225\345\210\233\345\273\272\346\226\207\344\273\266");
        return;
    }
    
    QTextStream out(&file);
    out << "file_path,score,result\n";
    
    for (const ImageInfo& info : images) {
        if (info.processed) {
            out << info.filePath << "," 
                << QString::number(info.confidence, 'f', 6) << ","
                << (info.isPositive ? "True" : "False") << "\n";
        }
    }
    
    QMessageBox::information(this, "\345\256\214\346\210\220", "\347\273\223\346\236\234\345\267\262\345\257\274\345\207\272\345\210\260\72\40" + fileName);
}

QPixmap NLKillerGUI::loadImageAsPixmap(const QString& path) {
    // ����Qt��ICC����
    qputenv("QT_LOGGING_RULES", "qt.gui.icc.debug=false");
    
    QPixmap pixmap;
    if (!pixmap.load(path)) {
        // ���ֱ�Ӽ���ʧ�ܣ������ȼ���ΪQImage��ת��
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
        statisticsLabel->setText("\347\273\237\350\256\241\72\40\346\234\252\345\212\240\350\275\275\345\233\276\345\203\217");
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
    QString statsText = QString("\346\234\252\346\211\253\346\217\217 %1 | \351\200\232\350\277\207 %2 | \345\217\221\347\216\260\345\245\266\351\276\231 %3").arg(unscanned).arg(passed).arg(detected);

    statisticsLabel->setText(statsText);
}
