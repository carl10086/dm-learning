package com.ysz.dm.logging;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

@SpringBootApplication
public class LogMain {

  public static void main(String[] args) throws Exception {
    ConfigurableApplicationContext context = SpringApplication.run(LogMain.class, args);
    SpringBean bean = context.getBean(SpringBean.class);
    bean.hello();
  }

}
