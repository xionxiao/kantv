package com.kantvai.kantvplayer.player.common.adapter;

import androidx.annotation.LayoutRes;
import androidx.annotation.Nullable;

import com.chad.library.adapter.base.BaseQuickAdapter;
import com.chad.library.adapter.base.BaseViewHolder;
import com.kantvai.kantvplayer.R;


import java.util.List;

import kantvai.media.player.bean.TrackInfoBean;


public class StreamAdapter extends BaseQuickAdapter<TrackInfoBean, BaseViewHolder> {

    public StreamAdapter(@LayoutRes int layoutResId, @Nullable List<TrackInfoBean> data) {
        super(layoutResId, data);
    }

    @Override
    public void convert(BaseViewHolder helper, TrackInfoBean item) {
        helper.setText(R.id.track_name_tv, item.getName())
                .setChecked(R.id.track_select_cb, item.isSelect())
                .addOnClickListener(R.id.track_ll)
                .addOnClickListener(R.id.track_select_cb);
    }
}
