package com.ysz.arm.blog.server.monitor;

import com.linecorp.armeria.common.HttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.logging.RequestLog;
import com.linecorp.armeria.internal.common.RequestContextExtension;
import com.linecorp.armeria.server.HttpService;
import com.linecorp.armeria.server.ServiceRequestContext;
import com.linecorp.armeria.server.SimpleDecoratingHttpService;
import java.util.function.Consumer;
import java.util.function.Supplier;
import lombok.extern.slf4j.Slf4j;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
@Slf4j
public final class MonitorService extends SimpleDecoratingHttpService {

  /**
   * Creates a new instance that decorates the specified {@link HttpService}.
   */
  public MonitorService(HttpService delegate) {
    super(delegate);
  }

//  public static Function<? extends HttpService, MonitorService> newDecorator() {
//    return (Function<HttpService, MonitorService>) MonitorService::new;
//  }

  @Override
  public HttpResponse serve(ServiceRequestContext ctx, HttpRequest req) throws Exception {
    log.info("start serve");
    long start = System.currentTimeMillis();
    HttpResponse resp = unwrap().serve(ctx, req);
    RequestContextExtension ctxExtension = ctx.as(RequestContextExtension.class);
    if (ctxExtension != null) {
      MonitorCtx monitorCtx = new MonitorCtx();
      ctxExtension.hook(() -> monitorCtx);
    }
//    ctx.log().whenAvailable().thenAccept(new Consumer<RequestLog>() {
//      @Override
//      public void accept(RequestLog requestLog) {
//        log.info("accept");
//      }
//    });
    long cost = System.currentTimeMillis() - start;
    log.info("cost:{}", cost);
    return resp;
  }
}
