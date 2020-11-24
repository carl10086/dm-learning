package com.ysz.dm.dubbo.echo.base.consumer;

import com.ysz.dm.dubbo.echo.base.api.DemoService;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.ReferenceConfig;
import org.apache.dubbo.config.RegistryConfig;
import org.apache.dubbo.config.bootstrap.DubboBootstrap;
import org.apache.dubbo.config.utils.ReferenceConfigCache;
import org.apache.dubbo.rpc.service.GenericService;

public class ConsumerApplication {

  public static void main(String[] args) {
    if (isClassic(args)) {
      runWithRefer();
    } else {
      runWithBootstrap();
    }
  }

  private static boolean isClassic(String[] args) {
    return args.length > 0 && "classic".equalsIgnoreCase(args[0]);
  }

  private static void runWithBootstrap() {
    ReferenceConfig<DemoService> reference = new ReferenceConfig<>();
    reference.setInterface(DemoService.class);
    reference.setGeneric("true");

    DubboBootstrap bootstrap = DubboBootstrap.getInstance();
    bootstrap.application(new ApplicationConfig("dubbo-demo-api-consumer"))
        .registry(new RegistryConfig("multicast://224.5.6.7:1234"))
        .reference(reference)
        .start();

    DemoService demoService = ReferenceConfigCache.getCache().get(reference);
    String message = demoService.sayHello("dubbo");
    System.out.println(message);

    // generic invoke
//    GenericService genericService = (GenericService) demoService;
//    Object genericInvokeResult = genericService
//        .$invoke("sayHello", new String[]{String.class.getName()},
//            new Object[]{"dubbo generic invoke"});
//    System.out.println(genericInvokeResult);
  }

  private static void runWithRefer() {
    ReferenceConfig<DemoService> reference = new ReferenceConfig<>();
    reference.setApplication(new ApplicationConfig("dubbo-demo-api-consumer"));
    reference.setRegistry(new RegistryConfig("multicast://224.5.6.7:1234"));
    reference.setInterface(DemoService.class);
    DemoService service = reference.get();
    String message = service.sayHello("dubbo");
    System.out.println(message);
  }
}
