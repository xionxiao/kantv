package com.kantvai.kantvplayer.bean;


public class ScanFolderBean {
    private String folder;
    private boolean isCheck;

    public ScanFolderBean(String folder, boolean isCheck) {
        this.folder = folder;
        this.isCheck = isCheck;
    }

    public String getFolder() {
        return folder;
    }

    public void setFolder(String folder) {
        this.folder = folder;
    }

    public boolean isCheck() {
        return isCheck;
    }

    public void setCheck(boolean check) {
        isCheck = check;
    }
}
