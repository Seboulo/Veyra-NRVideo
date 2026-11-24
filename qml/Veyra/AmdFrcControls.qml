import QtQuick
import QtQuick.Layouts

// AMD FRC options: shown only while AMD FRC is the frame generation method.
ColumnLayout {
    id: root
    property string namePrefix: "list"
    visible: veyra.fgBackendName === "amdfrc"
    spacing: 0
    Layout.fillWidth: true
    VRow {
        objectName: root.namePrefix + "-amd-frc-profile"
        label: qsTr("AMD FRC 运动搜索")
        hint: qsTr("自动：独显及 1440p 以上用 SUPER，其他用 HIGH。SUPER 的分层运动搜索层级更多，更跟得住快速转镜，GPU 占用稍高")
        VSelect {
            title: qsTr("AMD FRC 运动搜索")
            options: [
                {id:"0", label:qsTr("自动")},
                {id:"1", label:"LOW"},
                {id:"2", label:"HIGH"},
                {id:"3", label:"SUPER"}
            ]
            value: (options[veyra.amdFrcProfile] || options[0]).label
            onPicked: id => veyra.amdFrcProfile = Number(id)
        }
    }
    VRow {
        objectName: root.namePrefix + "-amd-frc-blend"
        label: qsTr("AMD FRC 不确定时混合")
        hint: qsTr("插帧把握不足时的处理。开（默认）：混合前后两帧，可能看到重影。关：重复一张真实帧，没有重影，但会短暂顿一下")
        VSwitch {
            objectName: root.namePrefix + "-amd-frc-blend-switch"
            checked: veyra.amdFrcBlend
            onToggled: checked => veyra.amdFrcBlend = checked
        }
    }
}
