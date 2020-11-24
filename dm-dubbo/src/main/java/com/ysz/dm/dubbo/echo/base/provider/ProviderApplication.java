package com.ysz.dm.dubbo.echo.base.provider;

import com.ysz.dm.dubbo.echo.base.api.DemoService;
import java.util.concurrent.CountDownLatch;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.RegistryConfig;
import org.apache.dubbo.config.ServiceConfig;
import org.apache.dubbo.config.bootstrap.DubboBootstrap;

public class ProviderApplication {

  public static void main(String[] args) throws Exception {
    if (isClassic(args)) {
      startWithExport();
    } else {
      startWithBootstrap();
    }
  }

  private static boolean isClassic(String[] args) {
    return args.length > 0 && "classic".equalsIgnoreCase(args[0]);
  }

  private static void startWithBootstrap() {
    ServiceConfig<DemoServiceImpl> service = new ServiceConfig<>();
    service.setInterface(DemoService.class);
    service.setRef(new DemoServiceImpl());

    DubboBootstrap bootstrap = DubboBootstrap.getInstance();
    bootstrap.application(new ApplicationConfig("dubbo-demo-api-provider"))
        .registry(new RegistryConfig("multicast://224.5.6.7:1234"))
        .service(service)
        .start()
        .await();
  }

  private static void startWithExport() throws InterruptedException {
    ServiceConfig<DemoServiceImpl> service = new ServiceConfig<>();
    service.setInterface(DemoService.class);
    service.setRef(new DemoServiceImpl());
    service.setApplication(new ApplicationConfig("dubbo-demo-api-provider"));
    service.setRegistry(new RegistryConfig("multicast://224.5.6.7:1234"));
    service.export();

    System.out.println("dubbo service started");
    new CountDownLatch(1).await();
  }
}
