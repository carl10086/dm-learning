package com.ysz.dm.fast.basic.juc.cas;

import java.util.concurrent.atomic.AtomicReference;

public class AtomicReferenceDm {

  private static class State {

    private final String name;

    private State(final String name) {
      this.name = name;
    }

    @Override
    public String toString() {
      final StringBuilder sb = new StringBuilder("State{");
      sb.append("name='").append(name).append('\'');
      sb.append('}');
      return sb.toString();
    }
  }

  public static void main(String[] args) throws Exception {
    AtomicReference<State> reference = new AtomicReference<>();

    final State old = new State("111");
    reference.set(old);

    final boolean b = reference.compareAndSet(old, new State("2222"));
    System.err.println(b);
  }

}
