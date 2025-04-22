package com.kantvai.kantvplayer.bean.event;

import java.io.Serializable;


public class SaveCurrentEvent implements Serializable {
    private String folderPath;
    private String videoPath;
    private long currentPosition;

    public SaveCurrentEvent() {
    }

    public SaveCurrentEvent(String folderPath, String videoPath, int currentPosition) {
        this.folderPath = folderPath;
        this.videoPath = videoPath;
        this.currentPosition = currentPosition;
    }

    public String getFolderPath() {
        return folderPath;
    }

    public void setFolderPath(String folderPath) {
        this.folderPath = folderPath;
    }

    public String getVideoPath() {
        return videoPath;
    }

    public void setVideoPath(String videoPath) {
        this.videoPath = videoPath;
    }

    public long getCurrentPosition() {
        return currentPosition;
    }

    public void setCurrentPosition(long currentPosition) {
        this.currentPosition = currentPosition;
    }
}
