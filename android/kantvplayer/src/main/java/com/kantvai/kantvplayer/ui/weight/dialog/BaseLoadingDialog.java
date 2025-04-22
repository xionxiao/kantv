package com.kantvai.kantvplayer.ui.weight.dialog;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.widget.TextView;

import com.blankj.utilcode.util.StringUtils;
import com.kantvai.kantvplayer.R;

public class BaseLoadingDialog extends Dialog {
    private String msg;

    public BaseLoadingDialog(Context context) {
        super(context, R.style.Dialog);
    }

    public BaseLoadingDialog(Context context, String msg) {
        super(context, R.style.Dialog);
        this.msg = msg;
    }

    public BaseLoadingDialog(Context context, int theme) {
        super(context, theme);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.dialog_base_loading);

        if (!StringUtils.isEmpty(msg)){
            TextView textView = this.findViewById(R.id.msg_tv);
            this.setCancelable(false);
            textView.setText(msg);
        }
    }
}
