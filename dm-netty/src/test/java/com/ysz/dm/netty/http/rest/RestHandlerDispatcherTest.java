package com.ysz.dm.netty.http.rest;

import com.ysz.dm.netty.http.rest.RestRequest.Method;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;

public class RestHandlerDispatcherTest {

  private RestHandlerDispatcher instance = RestHandlerDispatcher.getInstance();

  @Before
  public void setUp() throws Exception {
    instance.registerHandler(Method.GET, "/tst/users", new RestHandler() {
      @Override
      public String toString() {
        return "->/tst/users";
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
        mHandler = Optional.ofNullable(it.next()).flatMap(mh -> mh.getHandler(requestMethod));
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
    restRequest.setRawPath("/tst/users");
    return restRequest;
  }


}