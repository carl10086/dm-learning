package com.ysz.dm.ddd.vshop.domain.core.common.lang;

import com.google.common.collect.ImmutableMap;
import java.util.HashMap;
import java.util.Map;

/**
 * map builder for 链式编程
 *
 * @author carl
 * @create 2022-09-05 6:49 PM
 **/
public class MapBuilder<K, V> {

  private Map<K, V> builder;

  /**
   * 是否过滤掉 null key
   */
  private boolean ignoreNullKey;

  /**
   * 是否过滤掉 null value
   */
  private boolean ignoreNullValue;


  public MapBuilder() {
    this(true, true, 16, 0.75f, Provider.hashmap);
  }

  public MapBuilder(
      boolean ignoreNullKey,
      boolean ignoreNullValue,
      int initialCapacity,
      float loadFactor,
      Provider provider
  ) {
    this.ignoreNullKey = ignoreNullKey;
    this.ignoreNullValue = ignoreNullValue;
    this.builder = new HashMap<>(initialCapacity, loadFactor);
  }


  public MapBuilder<K, V> put(K k, V v) {
    if ((ignoreNullKey && k == null)
        ||
        (ignoreNullValue && (v == null))
    ) {
      return this;
    }
    this.builder.put(k, v);
    return this;
  }


  public MapBuilder<K, V> remove(K k) {
    this.builder.remove(k);
    return this;
  }


  public ImmutableMap<K, V> build() {
    return ImmutableMap.copyOf(this.builder);
  }


  private enum Provider {
    hashmap,
    treemap
  }
}
