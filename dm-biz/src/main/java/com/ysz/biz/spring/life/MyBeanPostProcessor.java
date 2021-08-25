package com.ysz.biz.spring.life;

import java.util.concurrent.ConcurrentHashMap;
import lombok.extern.slf4j.Slf4j;
import org.springframework.aop.support.AopUtils;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class MyBeanPostProcessor implements BeanPostProcessor {

  private ConcurrentHashMap<String, Long> startMap = new ConcurrentHashMap<>(256);
  private ConcurrentHashMap<String, Long> endMap = new ConcurrentHashMap<>(256);

  public MyBeanPostProcessor() {
    log.info("MyBeanPostProcessor constructed");
  }

  @Override
  public Object postProcessBeforeInitialization(Object bean, String beanName)
      throws BeansException {
    if (AopUtils.getTargetClass(bean) == LifeBean.class) {
      log.info("postProcessBeforeInitialization:{}", beanName);
    }
    startMap.put(beanName, System.currentTimeMillis());
    return bean;
  }

  @Override
  public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
    if (AopUtils.getTargetClass(bean) == LifeBean.class) {
      log.info("postProcessAfterInitialization:{}", beanName);
    }
    endMap.put(beanName, System.currentTimeMillis());
    return bean;
  }
}
