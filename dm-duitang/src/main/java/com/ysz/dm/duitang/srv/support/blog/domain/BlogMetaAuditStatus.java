package com.ysz.dm.duitang.srv.support.blog.domain;

public enum BlogMetaAuditStatus {
  blocked,
  locked,
  suspect,
  waiting_audit,
  normal;

  public boolean canBeForward() {
    return this == waiting_audit || this == normal;
  }
}
