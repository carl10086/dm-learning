package com.duitang.dm.learning.ddd.gw.domain.core.user;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class User {

  private UserCore core;
  private UserCnt cnt;

  public User setCore(UserCore core) {
    this.core = core;
    return this;
  }

  public User setCnt(UserCnt cnt) {
    this.cnt = cnt;
    return this;
  }
}
