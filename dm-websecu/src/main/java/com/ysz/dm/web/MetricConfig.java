package com.ysz.dm.web;

import io.micrometer.prometheus.PrometheusMeterRegistry;
import java.io.IOException;
import java.nio.charset.Charset;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
@Controller
@Configuration
@RequestMapping("/")
public class MetricConfig {


  @RequestMapping("metrics")
  public void prometheus(HttpServletResponse response) throws IOException {
    PrometheusMeterRegistry prometheusRegistry = PrometheusMeterRegistryHolder.getInstance();
    ServletOutputStream outputStream = response.getOutputStream();
    outputStream.write(prometheusRegistry.scrape().getBytes(Charset.defaultCharset()));
    outputStream.flush();
    outputStream.close();
  }

}
