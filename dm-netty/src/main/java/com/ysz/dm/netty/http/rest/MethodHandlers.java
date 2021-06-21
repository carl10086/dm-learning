package com.ysz.dm.netty.http.rest;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class MethodHandlers {

  private final String path;
  private final Map<RestRequest.Method, RestHandler> methodHandlers;

  MethodHandlers(String path, RestHandler handler, RestRequest.Method... methods) {
    this.path = path;
    this.methodHandlers = new HashMap<>(methods.length);
    for (RestRequest.Method method : methods) {
      methodHandlers.put(method, handler);
    }
  }

  /**
   * Add an additional method and handler for an existing path. Note that {@code MethodHandlers}
   * does not allow replacing the handler for an already existing method.
   */
  public MethodHandlers addMethod(RestRequest.Method method, RestHandler handler) {
    RestHandler existing = methodHandlers.putIfAbsent(method, handler);
    if (existing != null) {
      throw new IllegalArgumentException(
          "Cannot replace existing handler for [" + path + "] for method: " + method);
    }
    return this;
  }

  /**
   * Add a handler for an additional array of methods. Note that {@code MethodHandlers}
   * does not allow replacing the handler for an already existing method.
   */
  public MethodHandlers addMethods(RestHandler handler, RestRequest.Method... methods) {
    for (RestRequest.Method method : methods) {
      addMethod(method, handler);
    }
    return this;
  }

  /**
   * Return an Optional-wrapped handler for a method, or an empty optional if
   * there is no handler.
   */
  public Optional<RestHandler> getHandler(RestRequest.Method method) {
    return Optional.ofNullable(methodHandlers.get(method));
  }

  /**
   * Return a set of all valid HTTP methods for the particular path
   */
  public Set<RestRequest.Method> getValidMethods() {
    return methodHandlers.keySet();
  }

}
