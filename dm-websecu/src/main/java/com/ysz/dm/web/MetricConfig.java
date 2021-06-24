package com.ysz.dm.web;

import com.ysz.dm.web.metrics.example.ExampleGaugeReporter;
import io.micrometer.core.instrument.Tag;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.function.ToDoubleFunction;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
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
public class MetricConfig implements InitializingBean, DisposableBean {

  private PrometheusMeterRegistry prometheusRegistry = PrometheusMeterRegistryHolder.getInstance();

  private ExampleGaugeReporter exampleGaugeReporter;


  @RequestMapping("metrics")
  public void prometheus(HttpServletResponse response) throws IOException {
    ServletOutputStream outputStream = response.getOutputStream();
    outputStream.write(prometheusRegistry.scrape().getBytes(Charset.defaultCharset()));
    outputStream.flush();
    outputStream.close();
  }

  @Override
  public void afterPropertiesSet() throws Exception {
    this.exampleGaugeReporter = new ExampleGaugeReporter();

    final String name = "kafkaOffset";
    final Iterable<Tag> tags = Collections.emptyList();
    prometheusRegistry.gauge(name, tags,
        (ToDoubleFunction<Iterable<Tag>>) value -> exampleGaugeReporter.report());
  }

  @Override
  public void destroy() throws Exception {
    if (this.exampleGaugeReporter != null) {
      this.exampleGaugeReporter.close();
    }
  }
}
