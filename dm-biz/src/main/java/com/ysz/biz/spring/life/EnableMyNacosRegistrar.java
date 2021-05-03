package com.ysz.biz.spring.life;

import com.alibaba.nacos.spring.util.NacosBeanUtils;
import com.ysz.biz.spring.life.anno.EnableMyNacosAnno;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.BeanFactoryAware;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.context.EnvironmentAware;
import org.springframework.context.annotation.ImportBeanDefinitionRegistrar;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;
import org.springframework.core.annotation.AnnotationAttributes;
import org.springframework.core.env.Environment;
import org.springframework.core.type.AnnotationMetadata;

@Slf4j
public class EnableMyNacosRegistrar implements ImportBeanDefinitionRegistrar, EnvironmentAware,
    BeanFactoryAware {

  private Environment environment;

  private BeanFactory beanFactory;


  @Override
  public void setBeanFactory(BeanFactory beanFactory) throws BeansException {
    this.beanFactory = beanFactory;
  }

  @Override
  public void setEnvironment(Environment environment) {
    this.environment = environment;
  }

  @Override
  public void registerBeanDefinitions(AnnotationMetadata importingClassMetadata,
      BeanDefinitionRegistry registry) {
    log.info("start nacos registerBeanDefinitions");
    BeanDefinition annotationProcessor = BeanDefinitionBuilder
        .genericBeanDefinition(PropertySourcesPlaceholderConfigurer.class).getBeanDefinition();
    registry.registerBeanDefinition(PropertySourcesPlaceholderConfigurer.class.getName(),
        annotationProcessor);
    AnnotationAttributes attributes = AnnotationAttributes
        .fromMap(importingClassMetadata.getAnnotationAttributes(EnableMyNacosAnno.class.getName()));
    AnnotationAttributes globalProperties = (AnnotationAttributes) attributes
        .get("globalProperties");
    String dt_app_env = this.environment.getProperty("DT_APP_ENV");
    String namespace = environment.getProperty("DT_APP_NACOS_NS");
    System.err.println(dt_app_env);
    globalProperties.put("namespace", namespace);
    /*1. Register Global Nacos Properties Bean 注册全局的 nacos properties bean*/
    NacosBeanUtils.registerGlobalNacosProperties(attributes, registry, this.environment,
        "globalNacosProperties");
    /*2. 注册 nacos 通用的 bean*/
    NacosBeanUtils.registerNacosCommonBeans(registry);
    /*3. 注册 配置中心需要的 bean*/
    NacosBeanUtils.registerNacosConfigBeans(registry, this.environment, beanFactory);

    /*4. 注册 nacos 服务发现功能需要的 bean*/
//    NacosBeanUtils.registerNacosDiscoveryBeans(registry);

    /*5. 提前调用 FactoryBeanPostProcessor 方法、让下面 bean 可以直接使用 environment bean ?*/
    NacosBeanUtils.invokeNacosPropertySourcePostProcessor(beanFactory);
    log.info("finish nacos registerBeanDefinitions");
  }


}
