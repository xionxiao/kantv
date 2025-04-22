package com.kantvai.kantvplayer.bean;

import com.kantvai.kantvplayer.bean.params.HistoryParam;
import com.kantvai.kantvplayer.utils.net.CommJsonEntity;
import com.kantvai.kantvplayer.utils.net.CommJsonObserver;
import com.kantvai.kantvplayer.utils.net.NetworkConsumer;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;


public class PlayHistoryBean extends CommJsonEntity implements Serializable {

    private List<PlayHistoryAnimesBean> playHistoryAnimes;

    public List<PlayHistoryAnimesBean> getPlayHistoryAnimes() {
        return playHistoryAnimes;
    }

    public void setPlayHistoryAnimes(List<PlayHistoryAnimesBean> playHistoryAnimes) {
        this.playHistoryAnimes = playHistoryAnimes;
    }

    public static class PlayHistoryAnimesBean implements Serializable{
        private int animeId;
        private String animeTitle;
        private String type;
        private String typeDescription;
        private String imageUrl;
        private boolean isOnAir;
        private List<EpisodesBean> episodes;

        public int getAnimeId() {
            return animeId;
        }

        public void setAnimeId(int animeId) {
            this.animeId = animeId;
        }

        public String getAnimeTitle() {
            return animeTitle;
        }

        public void setAnimeTitle(String animeTitle) {
            this.animeTitle = animeTitle;
        }

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public String getTypeDescription() {
            return typeDescription;
        }

        public void setTypeDescription(String typeDescription) {
            this.typeDescription = typeDescription;
        }

        public String getImageUrl() {
            return imageUrl;
        }

        public void setImageUrl(String imageUrl) {
            this.imageUrl = imageUrl;
        }

        public boolean isIsOnAir() {
            return isOnAir;
        }

        public void setIsOnAir(boolean isOnAir) {
            this.isOnAir = isOnAir;
        }

        public List<EpisodesBean> getEpisodes() {
            return episodes;
        }

        public void setEpisodes(List<EpisodesBean> episodes) {
            this.episodes = episodes;
        }

        public static class EpisodesBean implements Serializable {
            private int episodeId;
            private String episodeTitle;
            private String lastWatched;
            private String airDate;

            public int getEpisodeId() {
                return episodeId;
            }

            public void setEpisodeId(int episodeId) {
                this.episodeId = episodeId;
            }

            public String getEpisodeTitle() {
                return episodeTitle;
            }

            public void setEpisodeTitle(String episodeTitle) {
                this.episodeTitle = episodeTitle;
            }

            public String getLastWatched() {
                return lastWatched;
            }

            public void setLastWatched(String lastWatched) {
                this.lastWatched = lastWatched;
            }

            public String getAirDate() {
                return airDate;
            }

            public void setAirDate(String airDate) {
                this.airDate = airDate;
            }
        }
    }

    public static void getPlayHistory(CommJsonObserver<PlayHistoryBean> observer, NetworkConsumer consumer){

    }

    public static void addPlayHistory(int episodeId, CommJsonObserver<CommJsonEntity> observer, NetworkConsumer consumer){
        HistoryParam historyParam = new HistoryParam();
        historyParam.setEpisodeIdList(new ArrayList<>());
        historyParam.getEpisodeIdList().add(episodeId);

    }
}
