#pragma once

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QScrollArea>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QThread>
#include <QtCore/QTimer>
#include <QtCore/QMutex>
#include <QtGui/QPixmap>
#include <QtGui/QKeyEvent>
#include <QtGui/QImage>
#include <vector>
#include <memory>
#include <future>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <mutex>

// ǰ������������ֱ�Ӱ���NLKillerͷ�ļ�
class NLKiller;
template<typename T, int dim> class YTensor;
using uchar = unsigned char;

// ͼ����Ϣ�ṹ��
struct ImageInfo {
    QString filePath;
    QString fileName;
    bool processed;
    float confidence;
    bool isPositive;
    
    ImageInfo() : processed(false), confidence(0.0f), isPositive(false) {}
};

// �첽�������߳�
class InferenceWorker : public QObject {
    Q_OBJECT

public:
    InferenceWorker(NLKiller* killer, std::vector<ImageInfo>* images);
    ~InferenceWorker();
    
    static YTensor<unsigned char, 3> loadImageToTensor(const QString& path);
    
public slots:
    void processImages();
    void processSingleImage(int index);

signals:
    void imageProcessed(int index, float confidence);
    void allImagesProcessed();
    void singleImageProcessed(int index, float confidence);
    void progressUpdated(int current, int total, float speed);

private:
    // �򻯵Ľ������ṹ
    struct DecodedResult {
        int index;
        std::shared_ptr<YTensor<unsigned char, 3>> tensor;
        
        DecodedResult(int idx, std::shared_ptr<YTensor<unsigned char, 3>> t) 
            : index(idx), tensor(t) {}
    };
    
    NLKiller* killer;
    std::vector<ImageInfo>* images;
    
    // ���߳̽��� + ���߳�����
    std::atomic<bool> shouldStop;
    std::queue<DecodedResult> decodedQueue;  // ������ɵ�ͼƬ����
    std::mutex queueMutex;
    std::vector<std::thread> decoderThreads;
    
    // ����
    void decoderWorker(int startIndex, int endIndex);
    void inferenceLoop();
};

class NLKillerGUI : public QMainWindow {
    Q_OBJECT

public:
    explicit NLKillerGUI(QWidget *parent = nullptr);
    ~NLKillerGUI();

protected:
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void openFolder();
    void batchInference();
    void refreshResults();
    void onConfidenceChanged(int value);
    void onImageProcessed(int index, float confidence);
    void onAllImagesProcessed();
    void onSingleImageProcessed(int index, float confidence);
    void onModelChanged(int index);
    void onTableSelectionChanged();
    void onProgressUpdated(int current, int total, float speed);
    void exportResults();

private:
    // UI���
    QWidget* centralWidget;
    QHBoxLayout* mainLayout;
    
    // ���ͼ��Ԥ��
    QScrollArea* imageScrollArea;
    QLabel* imageLabel;
    
    // �Ҳ�������
    QWidget* controlPanel;
    QVBoxLayout* controlLayout;
    
    QPushButton* openFolderBtn;
    QPushButton* batchInferenceBtn;
    
    QProgressBar* progressBar;
    QLabel* speedLabel;
    QWidget* progressWidget;
    QHBoxLayout* progressLayout;
    
    QLabel* confidenceLabel;
    QSlider* confidenceSlider;
    QLabel* confidenceValueLabel;
    
    QPushButton* refreshBtn;
    
    QTableWidget* imageTable;
    
    QLabel* currentImageStatusLabel;
    
    QLabel* statisticsLabel;
    
    QPushButton* exportBtn;
    
    QLabel* modelLabel;
    QComboBox* modelComboBox;
    
    // ����
    std::vector<ImageInfo> images;
    int currentImageIndex;
    float confidenceThreshold;
    
    // AI����
    std::unique_ptr<NLKiller> killer;
    QThread* workerThread;
    InferenceWorker* worker;
    
    // ֧�ֵ�ͼ���ʽ
    QStringList supportedFormats;
    
    // ����
    void setupUI();
    void setupImagePreview();
    void setupControlPanel();
    void loadImagesFromFolder(const QString& folderPath);
    void updateImagePreview();
    void updateImageTable();
    void updateCurrentImageStatus();
    void updateStatistics();
    void switchToImage(int index);
    void processCurrentImage();
    void loadModel(const QString& modelPath);
    QPixmap loadImageAsPixmap(const QString& path);
    
    // ģ��·��
    QStringList modelPaths;
};