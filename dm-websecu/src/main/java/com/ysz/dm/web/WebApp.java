package com.ysz.dm.web;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
@SpringBootApplication
@Configuration
@PropertySource(
    value = {
        "file:///Users/carl/Projects/IdeaProjects/dm-learning/dm-websecu/conf/a.properties"
    }

)
public class WebApp {

  public static void main(String[] args) throws Exception {
    SpringApplication.run(WebApp.class, args);
  }
}
