package com.ysz.dm.duitang.srv.support.blog.domain;

import lombok.Getter;

/**
 * 为了兼容老的 status、 把审核状态也留了下来 .
 */
public enum BlogForwardFinalStatus {

  deleted(5),
  normal(0),
  waiting_audit(7),
  blocked(6),
  suspect(10),
  locked(1);

  @Getter
  private final int val;

  BlogForwardFinalStatus(final int val) {
    this.val = val;
  }


  /**
   * 判断当前 blog status 作为 parent 是否可以转发
   * @return true: 可以，false: 不可以
   */
  public boolean canBeForward() {
    return this == waiting_audit || this == normal;
  }
}
