package com.ysz.dm.fast.http;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;

/**
 * @author carl
 */
public class Simple_Http_Dm_001 {

  public static void main(String[] args) throws Exception {
    URL url = new URL("http://www.163.com");
    HttpURLConnection connection = null;

    connection = (HttpURLConnection) url.openConnection();
    connection.setRequestMethod("GET");

    BufferedReader in = new BufferedReader(
        new InputStreamReader(connection.getInputStream(), Charset.forName("GBK")));
    String inputLine;
    StringBuffer content = new StringBuffer();
    while ((inputLine = in.readLine()) != null) {
      content.append(inputLine).append("\n");
    }
    in.close();
    System.out.println(content);
  }

}
