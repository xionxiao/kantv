package com.kantvai.kantvplayer.utils.net;

import com.blankj.utilcode.util.LogUtils;
import com.blankj.utilcode.util.NetworkUtils;

import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;

public class NetworkConsumer implements Consumer<Disposable> {

    private NetworkDisplay display;

    public NetworkConsumer() {

    }

    public NetworkConsumer(NetworkDisplay display) {
        this.display = display;
    }

    @Override
    public void accept(Disposable disposable) {
        if (NetworkUtils.isAvailableByPing()) {
            if (display != null) {
                display.normalNetwork();
            }
        } else {
            if (display != null) {
                display.noNetwork();
            }
            LogUtils.i("no network");
        }
    }

    interface NetworkDisplay {

        void normalNetwork();

        void noNetwork();
    }
}
