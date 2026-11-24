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
        hint: qsTr("自动：独显及 1440p 以上用 SUPER，其他用 HIGH。SUPER 跟得住快速转镜，GPU 占用稍高")
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
        hint: qsTr("开（默认）：运动不确定处混合前后两帧，可能出现重影。关：始终用运动补偿帧，减少重影但可能出现扭曲")
        VSwitch {
            objectName: root.namePrefix + "-amd-frc-blend-switch"
            checked: veyra.amdFrcBlend
            onToggled: checked => veyra.amdFrcBlend = checked
        }
    }
}
