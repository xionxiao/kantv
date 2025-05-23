 /*
  * Copyright (c) Project KanTV. 2021-2023
  *
  * Copyright (c) 2024- KanTV Authors
  *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
  * of this software and associated documentation files (the "Software"), to
  * deal in the Software without restriction, including without limitation the
  * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  * sell copies of the Software, and to permit persons to whom the Software is
  * furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in
  * all copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  * IN THE SOFTWARE.
  */
package kantvai.media.player;

import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.os.Parcel;
import android.os.Parcelable;


public class KANTVContentDescriptor implements Parcelable {
    private static final String TAG = KANTVContentDescriptor.class.getName();
    private String mName; // program name
    private String mURL;  // program url
    private KANTVUrlType mUrlType;
    private boolean mIsProtected; //DRM protected content
    private boolean mIsLive;      //Live or VOD
    private String mLicenseURL;   //DRM license
    private String mDrmScheme;    //DRM scheme

    private String mTVGID;        //tvg id;
    private String mTVGLogoURL;   //tvg logo URL
    private String mTVGLogoName;  //tvg logo name


    private String mCountry;
    private String mLanguage;
    private String mIdString;
    private String mLogo;       //TV logo if mMediaType == MEDIA_TV
    private String mPoster;     //Movie poster if mMediaType == MEDIA_MOVIE
    private String mGroupTitle; //category of content
    private KANTVMediaType mMediaType;

    public KANTVContentDescriptor(String name, String url, KANTVUrlType urlType) {
        this.mName = name;
        this.mURL = url;
        this.mUrlType = urlType;
    }

    public KANTVContentDescriptor(String tvglogoURL, String groupTitle, String name, String url) {
        this.mTVGLogoURL = tvglogoURL;
        this.mGroupTitle = groupTitle;
        this.mName = name;
        this.mURL = url;
        this.mUrlType = KANTVUrlType.HLS;
    }

    public KANTVContentDescriptor(String tvglogoURL, String tvglogoName, String groupTitle, String name, String url) {
        this.mTVGLogoURL = tvglogoURL;
        this.mTVGLogoName = tvglogoName;
        this.mGroupTitle = groupTitle;
        this.mName = name;
        this.mURL = url;
        this.mUrlType = KANTVUrlType.HLS;
    }

    public KANTVContentDescriptor(String name, String url, KANTVUrlType urlType, boolean isProtected, boolean isLive, String drmScheme, String licenseURL) {
        this.mName = name;
        this.mURL = url;
        //KANTVLog.j(TAG, "url : " + this.mURL);
        this.mUrlType = urlType;
        this.mIsProtected = isProtected;
        this.mIsLive      = isLive;
        this.mDrmScheme   = drmScheme;
        this.mLicenseURL = licenseURL;
    }

    public KANTVContentDescriptor(String name, String url, KANTVUrlType urlType, boolean isProtected, boolean isLive, String drmScheme, String licenseURL, String moviePoster) {
        this.mName = name;
        this.mURL = url;
        //KANTVLog.j(TAG, "url : " + this.mURL);
        this.mUrlType = urlType;
        this.mIsProtected = isProtected;
        this.mIsLive      = isLive;
        this.mDrmScheme   = drmScheme;
        this.mLicenseURL = licenseURL;
        this.mPoster = moviePoster;
    }

    public KANTVContentDescriptor(String name, String url, KANTVUrlType urlType, KANTVMediaType mediaType, boolean isProtected, boolean isLive, String drmScheme, String licenseURL) {
        this.mName = name;
        this.mURL = url;
        //KANTVLog.j(TAG, "url : " + this.mURL);
        this.mUrlType = urlType;
        this.mMediaType = mediaType;
        this.mIsProtected = isProtected;
        this.mIsLive      = isLive;
        this.mDrmScheme   = drmScheme;
        this.mLicenseURL = licenseURL;
    }

    public String getName() {
        return mName;
    }

    public String getLicenseURL() {
        return mLicenseURL;
    }

    public String getDrmScheme() {
        return mDrmScheme;
    }

    public String getUrl() {
        //KANTVLog.j(TAG, "url : " + this.mURL);
        return mURL;
    }

    public String getPoster() { return mPoster; }

    public KANTVUrlType getType() {
        switch (mUrlType) {
            case HLS:
                return KANTVUrlType.HLS;
            case RTMP:
                return KANTVUrlType.RTMP;
            case DASH:
                return KANTVUrlType.DASH;
            case MKV:
                return KANTVUrlType.MKV;
            case TS:
                return KANTVUrlType.TS;
            case MP4:
                return KANTVUrlType.MP4;
            case MP3:
                return KANTVUrlType.MP3;
            case OGG:
                return KANTVUrlType.OGG;
            case AAC:
                return KANTVUrlType.AAC;
            case AC3:
                return KANTVUrlType.AC3;

            case FILE:
            default:
                return KANTVUrlType.FILE;
        }
    }

    public boolean getIsProtected() {
        return mIsProtected;
    }

    public boolean getIsLive() {
        return mIsLive;
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel arg0, int arg1) {
        arg0.writeString(mName);
        arg0.writeString(mURL);
        arg0.writeValue(mUrlType);
        arg0.writeValue(Boolean.valueOf(mIsProtected));
        arg0.writeValue(Boolean.valueOf(mIsLive));
        arg0.writeString(mDrmScheme);
        arg0.writeString(mLicenseURL);
        arg0.writeString(mPoster);
    }

    public static final Parcelable.Creator<KANTVContentDescriptor> CREATOR = new Parcelable.Creator<KANTVContentDescriptor>() {
        public KANTVContentDescriptor createFromParcel(Parcel in) {
            return new KANTVContentDescriptor(in);
        }

        public KANTVContentDescriptor[] newArray(int size) {
            return new KANTVContentDescriptor[size];
        }
    };

    private KANTVContentDescriptor(Parcel in) {
        mName = in.readString();
        mURL = in.readString();
        mUrlType = (KANTVUrlType) in.readValue(null);
        mIsProtected = (Boolean) in.readValue(null);
        mIsLive = (Boolean) in.readValue(null);
        mDrmScheme = in.readString();
        mLicenseURL = in.readString();
        mPoster = in.readString();
    }

    public Drawable getDrawable(Resources res, int resID) {
        Drawable icon = null;
        if (mUrlType == KANTVUrlType.HLS) {
            //icon = res.getDrawable(R.drawable.hls);
            icon = res.getDrawable(resID);
        } else if (mUrlType == KANTVUrlType.RTMP) {
            //icon = res.getDrawable(R.drawable.rtmp);
            icon = res.getDrawable(resID);
        } else if (mUrlType == KANTVUrlType.DASH) {
            //icon = res.getDrawable(R.drawable.dash);
            icon = res.getDrawable(resID);
        } else if (mUrlType == KANTVUrlType.FILE) {
            //icon = res.getDrawable(R.drawable.dash);
            icon = res.getDrawable(resID);
        } else {
            KANTVLog.d(TAG, "unknown type:" + mUrlType.toString());
            //icon = res.getDrawable(R.drawable.file);
            icon = null;
        }
        if (icon == null) {
            KANTVLog.d(TAG, "can't get drawable, pls check");
        }
        return icon;
    }

    public static native int kantv_anti_remove_rename_this_file();
}