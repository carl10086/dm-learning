package com.ysz.dm.netty.http.rest;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class RestHandlerDispatcher {

  private final PathTrie<MethodHandlers> handlers = new PathTrie<>(RestUtils.REST_DECODER);

  public void registerHandler(
      RestRequest.Method method,
      String path,
      RestHandler handler
  ) {
    handlers.insertOrUpdate(
        path,
        new MethodHandlers(path, handler, method),
        (mHandlers, newMHandler) -> mHandlers.addMethods(handler, method));
  }

  public static RestHandlerDispatcher getInstance() {
    return SingletonHolder.instance;
  }

  private String getPath(RestRequest request) {
    // we use rawPath since we don't want to decode it while processing the path resolution
    // so we can handle things like:
    // my_index/my_type/http%3A%2F%2Fwww.google.com
    return request.rawPath();
  }


  Iterator<MethodHandlers> getAllHandlers(final RestRequest request) {
    // Between retrieving the correct path, we need to reset the parameters,
    // otherwise parameters are parsed out of the URI that aren't actually handled.
    final Map<String, String> originalParams = new HashMap<>(request.params());
    return handlers.retrieveAll(getPath(request), () -> {
      // PathTrie modifies the request, so reset the params between each iteration
      request.params().clear();
      request.params().putAll(originalParams);
      return request.params();
    });
  }


  private static class SingletonHolder {

    private static RestHandlerDispatcher instance = new RestHandlerDispatcher();
  }

}
