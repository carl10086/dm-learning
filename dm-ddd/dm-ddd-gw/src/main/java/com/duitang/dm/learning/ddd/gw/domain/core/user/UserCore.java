package com.duitang.dm.learning.ddd.gw.domain.core.user;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UserCore {

  private Long id;
  private String nickname;
  private String telephone;
  private Integer status;


  private Instant createAt;
  private Instant updateAt;

  public UserCore setId(Long id) {
    this.id = id;
    return this;
  }

  public UserCore setStatus(Integer status) {
    this.status = status;
    return this;
  }

  public UserCore setNickname(String nickname) {
    this.nickname = nickname;
    return this;
  }

  public UserCore setTelephone(String telephone) {
    this.telephone = telephone;
    return this;
  }

  public UserCore setCreateAt(Instant createAt) {
    this.createAt = createAt;
    return this;
  }

  public UserCore setUpdateAt(Instant updateAt) {
    this.updateAt = updateAt;
    return this;
  }
}
