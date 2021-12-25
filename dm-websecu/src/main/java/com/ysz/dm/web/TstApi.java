package com.ysz.dm.web;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import org.apache.commons.io.IOUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RequestMapping("/tst")
@RestController
public class TstApi {


  @RequestMapping("/")
  public String tst() throws Exception {
    URL url = new URL("http://10.200.64.5:8000/1.txt");
    final URLConnection urlConnection = url.openConnection();
    final InputStream inputStream = urlConnection.getInputStream();
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    IOUtils.copy(inputStream, outputStream);
    return outputStream.toString();
  }

}
