package com.ysz.dm.web;

import com.ysz.dm.web.filter.MicroMeterFilter;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationProperties;
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
  @ConfigurationProperties(prefix = "redis.feedrec")
  public FeedRecRedisProperties feedRecRedisProperties() {
    return new FeedRecRedisProperties();
  }

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
