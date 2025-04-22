package com.kantvai.kantvplayer.utils.interf;

import java.io.IOException;
import java.util.Map;

public interface Engine {

    String METHOD_GET = "GET";

    String METHOD_POST = "POST";

    String CHARSET_NAME = "UTF-8";

    String CONTENT_TYPE_LABEL = "Content-Type";

    String CONTENT_TYPE_VALUE_JSON = "application/json; charset=utf-8";

    String get(String url) throws Exception;

    String post(String method, Map<String, String> paramsMap) throws IOException;

    String post(String method, String[] paramKeys, String[] paramValues) throws IOException;

}
