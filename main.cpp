#include <QApplication>
#include "gui.hpp"

int main(int argc, char *argv[]){
    QApplication app(argc, argv);

    // 设置应用程序信息
    QApplication::setApplicationName("NLKiller GUI");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("NLKiller");

    // 创建并显示主窗口
    NLKillerGUI window;
    window.show();

    return app.exec();
}
