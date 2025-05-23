package kantvai.media.encoder;

import android.os.Handler;
import android.os.Message;

import java.lang.ref.WeakReference;

import kantvai.media.player.KANTVLog;


/**
 * Created by leo.ma
 */

public class SrsEncodeHandler extends Handler {

    private static final int MSG_ENCODE_NETWORK_WEAK = 0;
    private static final int MSG_ENCODE_NETWORK_RESUME = 1;
    private static final int MSG_ENCODE_ILLEGAL_ARGUMENT_EXCEPTION = 2;
    private static final String TAG = SrsEncodeHandler.class.getName();

    private WeakReference<SrsEncodeListener> mWeakListener;

    public SrsEncodeHandler(SrsEncodeListener listener) {
        KANTVLog.j(TAG, "here");
        mWeakListener = new WeakReference<>(listener);
    }

    public void notifyNetworkWeak() {
        sendEmptyMessage(MSG_ENCODE_NETWORK_WEAK);
    }

    public void notifyNetworkResume() {
        sendEmptyMessage(MSG_ENCODE_NETWORK_RESUME);
    }

    public void notifyEncodeIllegalArgumentException(IllegalArgumentException e) {
        obtainMessage(MSG_ENCODE_ILLEGAL_ARGUMENT_EXCEPTION, e).sendToTarget();
    }
    
    @Override  // runs on UI thread
    public void handleMessage(Message msg) {
        SrsEncodeListener listener = mWeakListener.get();
        if (listener == null) {
            return;
        }

        switch (msg.what) {
            case MSG_ENCODE_NETWORK_WEAK:
                listener.onNetworkWeak();
                break;
            case MSG_ENCODE_NETWORK_RESUME:
                listener.onNetworkResume();
                break;
            case MSG_ENCODE_ILLEGAL_ARGUMENT_EXCEPTION:
                listener.onEncodeIllegalArgumentException((IllegalArgumentException) msg.obj);
            default:
                throw new RuntimeException("unknown msg " + msg.what);
        }
    }

    public interface SrsEncodeListener {

        void onNetworkWeak();

        void onNetworkResume();

        void onEncodeIllegalArgumentException(IllegalArgumentException e);
    }
}
