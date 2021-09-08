package com.ysz.dm.netty.dm.http.rest;

import com.ysz.dm.netty.dm.http.rest.RestRequest.Method;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;

public class RestHandlerDispatcherTest {

  private RestHandlerDispatcher instance = RestHandlerDispatcher.getInstance();

  @Before
  public void setUp() throws Exception {
    instance.registerHandler(Method.GET, "//{}cateId/{}", new RestHandler() {
      @Override
      public String toString() {
        return "->//tst/{cateId}/users";
      }
    });

    instance.registerHandler(Method.GET, "/tst/abc/users", new RestHandler() {
      @Override
      public String toString() {
        return "/tst/abc/users";
      }
    });
  }

  @Test
  public void getAllHandlers() {
    boolean requestHandled = false;
    Iterator<MethodHandlers> allHandlers = instance.getAllHandlers(mock());
    RestRequest.Method requestMethod = Method.GET;
    for (Iterator<MethodHandlers> it = allHandlers; it.hasNext(); ) {
      Optional<RestHandler> mHandler = Optional.empty();
      if (requestMethod != null) {
        MethodHandlers next = it.next();
        mHandler = Optional.ofNullable(next).flatMap(mh -> mh.getHandler(requestMethod));
      }
      if (mHandler.isPresent()) {
        RestHandler restHandler = mHandler.get();
        System.err.println(restHandler);
      }
      if (requestHandled) {
        break;
      }
    }


  }


  public RestRequest mock() {
    RestRequest restRequest = new RestRequest();
    restRequest.setParams(new HashMap<>());
    restRequest.setRawPath("/tst/abc/users");
    return restRequest;
  }


}