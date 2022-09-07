package com.ysz.dm.biz.spring.properties;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.PropertySource;

/**
 * @author carl
 * @create 2022-09-07 1:05 PM
 **/
@Slf4j
@SpringBootApplication
@PropertySource("classpath:spring/custom.properties")
@EnableConfigurationProperties(User.class)
public class PropertiesMain {

  public static void main(String[] args) {
    ConfigurableApplicationContext run = SpringApplication.run(PropertiesMain.class);
    log.info("user:{}", run.getBean(User.class));
  }

}
