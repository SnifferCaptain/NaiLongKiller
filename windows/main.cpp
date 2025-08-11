#include <QApplication>
#include "gui.hpp"

int main(int argc, char *argv[]){
    QApplication app(argc, argv);

    // ����Ӧ�ó�����Ϣ
    QApplication::setApplicationName("NLKiller GUI");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("NLKiller");

    // ��������ʾ������
    NLKillerGUI window;
    window.show();

    return app.exec();
}
