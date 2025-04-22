package com.kantvai.kantvplayer.utils.net.utils;

import android.net.ParseException;

import com.google.gson.JsonParseException;
import com.google.gson.stream.MalformedJsonException;
import com.kantvai.kantvplayer.utils.net.gson.GsonFactory;

import org.apache.http.conn.ConnectTimeoutException;
import org.json.JSONException;

import java.io.Reader;
import java.net.ConnectException;

import okhttp3.ResponseBody;
import retrofit2.HttpException;


class ResponseErrorHandle {
    private static final int UNAUTHORIZED = 401;
    private static final int FORBIDDEN = 403;
    private static final int NOT_FOUND = 404;
    private static final int LOSS_PARAMS = 405;
    private static final int REQUEST_TIMEOUT = 408;
    private static final int INTERNAL_SERVER_ERROR = 500;
    private static final int SERVICE_UNAVAILABLE = 503;

    static ResponseError handleError(Throwable e) {
        ResponseError error;
        if (e instanceof HttpException) {
            HttpException exception = (HttpException) e;
            return handleHttpError(exception);
        } else if (e instanceof JsonParseException
                || e instanceof JSONException
                || e instanceof ParseException || e instanceof MalformedJsonException) {
            error = new ResponseError(e, ERROR.PARSE_ERROR);
            error.message = "解析响应数据错误";
            return error;
        } else if (e instanceof ConnectException) {
            error = new ResponseError(e, ERROR.NETWORK_ERROR);
            error.message = "连接失败，请重试";
            return error;
        } else if (e instanceof javax.net.ssl.SSLHandshakeException) {
            error = new ResponseError(e, ERROR.SSL_ERROR);
            error.message = "证书验证失败";
            return error;
        } else if (e instanceof ConnectTimeoutException) {
            error = new ResponseError(e, ERROR.TIMEOUT_ERROR);
            error.message = "连接超时，请重试";
            return error;
        } else if (e instanceof java.net.SocketTimeoutException) {
            error = new ResponseError(e, ERROR.TIMEOUT_ERROR);
            error.message = "请求超时，请重试";
            return error;
        } else {
            error = new ResponseError(e, ERROR.UNKNOWN);
            error.message = e.getMessage();
            return error;
        }
    }


    private static ResponseError handleHttpError(HttpException exception) {
        ResponseError error = new ResponseError(exception, exception.code());

        switch (exception.code()) {
            case UNAUTHORIZED:
                error.message = "操作未授权：401";
                break;
            case FORBIDDEN:
                error.message = "请求被拒绝：403";
                break;
            case NOT_FOUND:
                error.message = "资源不存在：404";
                break;
            case LOSS_PARAMS:
                error.message = "缺少参数：405";
                break;
            case REQUEST_TIMEOUT:
                error.message = "服务器执行超时：408";
                break;
            case INTERNAL_SERVER_ERROR:
                error.message = "服务器内部错误：500";
                break;
            case SERVICE_UNAVAILABLE:
                error.message = "服务器不可用：503";
                break;
            default:
                error.message = "网络错误：" + exception.code();
                break;
        }
        return error;
    }



    class ERROR {
        /**
         * 未知错误
         */
        static final int UNKNOWN = 1000;
        /**
         * 解析错误
         */
        static final int PARSE_ERROR = 1001;
        /**
         * 网络错误
         */
        static final int NETWORK_ERROR = 1002;
        /**
         * 协议出错
         */
        static final int HTTP_ERROR = 1003;

        /**
         * 证书出错
         */
        static final int SSL_ERROR = 1005;

        /**
         * 连接超时
         */
        static final int TIMEOUT_ERROR = 1006;
    }
}
