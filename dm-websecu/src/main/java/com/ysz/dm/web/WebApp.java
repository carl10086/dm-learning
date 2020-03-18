package com.ysz.dm.web;

import com.ysz.dm.web.filter.MicroMeterFilter;
import io.micrometer.core.instrument.binder.jvm.ClassLoaderMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmGcMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmThreadMetrics;
import io.micrometer.core.instrument.binder.logging.LogbackMetrics;
import io.micrometer.core.instrument.binder.system.FileDescriptorMetrics;
import io.micrometer.core.instrument.binder.system.ProcessorMetrics;
import io.micrometer.core.instrument.binder.system.UptimeMetrics;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import io.micrometer.prometheus.PrometheusRenameFilter;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
@SpringBootApplication
@Configuration
public class WebApp {


  @Bean
  public FilterRegistrationBean microMeterFilter() {
    FilterRegistrationBean registrationBean = new FilterRegistrationBean(new MicroMeterFilter());
    registrationBean.addUrlPatterns("/*");
    return registrationBean;
  }

  public static void main(String[] args) throws Exception {
    SpringApplication.run(WebApp.class, args);
  }
}
