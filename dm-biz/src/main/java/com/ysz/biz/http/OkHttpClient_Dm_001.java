package com.ysz.biz.http;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.Charset;
import okhttp3.Call;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

/**
 * @author carl
 */
public class OkHttpClient_Dm_001 {

  private ObjectMapper objectMapper = new ObjectMapper();

  public OkHttpClient okHttpClient = new OkHttpClient.Builder().build();

  public void post() throws Exception {
    String s = objectMapper.writeValueAsString(new GelfHttpData());
    System.out.println(s);

    final RequestBody requestBody = RequestBody
        .create(s.getBytes(Charset.defaultCharset()), MediaType.parse("application/json"));
    final Request request = new Request.Builder().url("http://10.1.2.5:12201/gelf")
        .post(requestBody)
        .build();
    final Call call = okHttpClient.newCall(request);
    String s1 = call.execute().toString();
    System.out.println(s1);

  }


  public static void main(String[] args) throws Exception {
    new OkHttpClient_Dm_001().post();
  }

}
