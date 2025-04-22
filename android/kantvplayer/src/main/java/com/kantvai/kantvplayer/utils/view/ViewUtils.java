package com.kantvai.kantvplayer.utils.view;

import androidx.core.view.ViewCompat;
import android.view.View;

public class ViewUtils {

    public static void doOnAttach(View view, ActionListener listener) {
        if (ViewCompat.isAttachedToWindow(view)) {
            listener.onAction(view);
        } else {
            view.addOnAttachStateChangeListener(new View.OnAttachStateChangeListener() {
                @Override
                public void onViewAttachedToWindow(View view) {
                    view.removeOnAttachStateChangeListener(this);
                    listener.onAction(view);
                }

                @Override
                @SuppressWarnings("PMD.UncommentedEmptyMethodBody")
                public void onViewDetachedFromWindow(View view) {}
            });
        }
    }

    public static void doOnDetach(View view, ActionListener listener) {
        if (!ViewCompat.isAttachedToWindow(view)) {
            listener.onAction(view);
        } else {
            view.addOnAttachStateChangeListener(new View.OnAttachStateChangeListener() {
                @Override
                @SuppressWarnings("PMD.UncommentedEmptyMethodBody")
                public void onViewAttachedToWindow(View view) {}

                @Override
                public void onViewDetachedFromWindow(View view) {
                    view.removeOnAttachStateChangeListener(this);
                    listener.onAction(view);
                }
            });
        }
    }

    public interface ActionListener {
        void onAction(View view);
    }
}
