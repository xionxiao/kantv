package com.kantvai.kantvplayer.base;

import android.app.Dialog;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.kantvai.kantvplayer.R;
import com.kantvai.kantvplayer.ui.weight.dialog.BaseLoadingDialog;
import com.kantvai.kantvplayer.utils.CommonUtils;
import com.kantvai.kantvplayer.utils.interf.presenter.BaseMvpPresenter;


public abstract class BaseMvpActivity<T extends BaseMvpPresenter> extends BaseAppCompatActivity {

    protected T presenter;
    protected Dialog dialog;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        presenter = initPresenter();
        super.onCreate(savedInstanceState);
    }

    @Override
    protected void init() {
        super.init();
        presenter.init();
    }

    @Override
    public void initPageView() {
        presenter.initPage();
    }

    @Override
    public void initPageViewListener() {

    }

    @Override
    protected void process(Bundle savedInstanceState) {
        super.process(savedInstanceState);
        presenter.process(savedInstanceState);
    }

    @Override
    protected void onResume() {
        super.onResume();
        presenter.resume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        presenter.pause();
    }

    @Override
    protected void onDestroy() {
        presenter.destroy();
        dialog = null;
        super.onDestroy();
    }

    @Override
    protected int getToolbarColor() {
        return CommonUtils.getResColor(R.color.theme_color);
    }

    @Override
    protected void setStatusBar() {
        super.setStatusBar();
    }

    public void showLoadingDialog() {
        if (dialog == null || !dialog.isShowing()) {
            dialog = new BaseLoadingDialog(this);
            dialog.show();
        }
    }

    public void showLoadingDialog(String msg) {
        if (dialog == null || !dialog.isShowing()) {
            dialog = new BaseLoadingDialog(this, msg);
            dialog.show();
        }
    }

    public void dismissLoadingDialog() {
        if (dialog != null && dialog.isShowing()) {
            dialog.dismiss();
            dialog = null;
        }
    }

    protected abstract @NonNull
    T initPresenter();
}
