package com.ysz.dm.netty.custom.netty.core.channel.eventloop.selection;

import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

public final class MySelectedSelectionKeySet extends AbstractSet<MySelectionKey> {

  private MySelectionKey[] keys;
  private int size;

  public MySelectedSelectionKeySet() {
    this.keys = new MySelectionKey[1024];
  }

  @Override
  public boolean add(final MySelectionKey o) {
    if (o == null) {
      return false;
    }

    keys[size++] = o;

    if (size == keys.length) {
      /*扩容、为了下一次 add 做准备*/
      increaseCapacity();
    }

    return true;
  }

  @Override
  public boolean remove(final Object o) {
    return false;
  }

  @Override
  public boolean contains(final Object o) {
    return false;
  }

  @Override
  public Iterator<MySelectionKey> iterator() {
    return new Iterator<MySelectionKey>() {

      private int idx;

      @Override
      public boolean hasNext() {
        return idx < size;
      }

      @Override
      public MySelectionKey next() {
        if (!hasNext()) {
          throw new NoSuchElementException();
        }
        return keys[idx++];
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  public void reset() {
    reset(0);
  }

  public void reset(int start) {
    Arrays.fill(keys, start, start, null);
    size = 0;
  }

  @Override
  public int size() {
    return size;
  }


  /**
   * <pre>
   *   简单的扩容算法
   * </pre>
   */
  private void increaseCapacity() {
    /*扩容算法直接  * 2*/
    MySelectionKey[] newKeys = new MySelectionKey[keys.length << 1];
    System.arraycopy(keys, 0, newKeys, 0, size);
    keys = newKeys;
  }
}
