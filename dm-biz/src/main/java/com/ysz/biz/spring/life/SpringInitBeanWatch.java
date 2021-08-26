package com.ysz.biz.spring.life;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanFactoryPostProcessor;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;
import org.springframework.context.ApplicationListener;
import org.springframework.context.event.ContextRefreshedEvent;

public class SpringInitBeanWatch implements BeanFactoryPostProcessor, BeanPostProcessor,
    ApplicationListener<ContextRefreshedEvent> {

  private long launchTime;
  private ConcurrentHashMap<String, Long> startMap = new ConcurrentHashMap<>(256);
  private ConcurrentHashMap<String, Long> endMap = new ConcurrentHashMap<>(256);

  @Override
  public void postProcessBeanFactory(final ConfigurableListableBeanFactory beanFactory)
      throws BeansException {
    this.launchTime = System.nanoTime();
  }

  @Override
  public Object postProcessBeforeInitialization(final Object bean, final String beanName)
      throws BeansException {
    startMap.put(beanName, System.nanoTime());
    return bean;
  }

  @Override
  public Object postProcessAfterInitialization(final Object bean, final String beanName)
      throws BeansException {
    endMap.put(beanName, System.nanoTime());
    return bean;
  }


  private void log() {
    System.err.println("finish log...");
    List<String> lines = new ArrayList<>(startMap.size() + 3);
    final long end = System.nanoTime();
    String fileName = "/duitang/logs/usr/saturn/spring_bean_init.log";
    /*第一行写 lanuchTime*/
    lines.add(launchTime + "");
    /*第二行写 end*/
    lines.add(end + "");
    /*第三行开始写入 startMap Size*/
    lines.add(startMap.size() + "");
    /*写入 startMap*/
    for (Entry<String, Long> entry : startMap.entrySet()) {
      lines.add(entry.getKey() + ":" + entry.getValue());
    }

    /*写入 endMap size*/
    lines.add(endMap.size() + "");
    for (Entry<String, Long> entry : endMap.entrySet()) {
      lines.add(entry.getKey() + ":" + entry.getValue());
    }
    try {
      FileUtils.writeLines(new File(fileName), lines);
    } catch (IOException ignored) {
    }
  }

  @Override
  public void onApplicationEvent(final ContextRefreshedEvent event) {
    log();
  }
}
