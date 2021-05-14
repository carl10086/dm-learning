package com.ysz.dm.logging.spring;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.PropertySource;

@SpringBootApplication
@PropertySource(value = "file:///Users/carl.yu/tmp/useless/application.properties")
public class LogMain {

  public static void main(String[] args) throws Exception {
    ConfigurableApplicationContext context = SpringApplication.run(LogMain.class, args);
    SpringBean bean = context.getBean(SpringBean.class);
    bean.hello();
  }

}
