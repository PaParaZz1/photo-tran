#include "st_soft_isp_utils.h"


static const char * sBuildDateTime = "2019-03-16 10:47:04";
static const char * sBuildCommitID = "97e3d1f3c51932252c29df80d3126878b76f18fd";
static const char * sBuildCommitMessage = "add align merge func";
static const char * sBuildGitUserName = "niuyazhe";
static const char * sBuildGituserEmail = "niuyazhe@sensetime.com";
static const char * sBuildMachineName = "niuyazhe-System-Product-Name";
static const char * sBuildPCUserName = "niuyazhe";
static const char * sBuildMachineIP = "10.1.31.189";
static const char * sBuildCodePath = "/home/niuyazhe/st_soft_isp";
void print_build_info()
{
    SOFT_ISP_LOGI("=================== st_soft_isp build info ===================");
    SOFT_ISP_LOGI("date time:      %s", sBuildDateTime);
    SOFT_ISP_LOGI("commit id:      %s", sBuildCommitID);
    SOFT_ISP_LOGI("commit message: %s", sBuildCommitMessage);
    SOFT_ISP_LOGI("git username:   %s", sBuildGitUserName);
    SOFT_ISP_LOGI("git user email: %s", sBuildGituserEmail);
    SOFT_ISP_LOGI("machine name:   %s", sBuildMachineName);
    SOFT_ISP_LOGI("pc username:    %s", sBuildPCUserName);
    SOFT_ISP_LOGI("machine ip:     %s", sBuildMachineIP);
    SOFT_ISP_LOGI("code path:      %s", sBuildCodePath);
    SOFT_ISP_LOGI("==============================================================");
}
