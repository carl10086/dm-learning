package com.duitang.dm.learning.ddd.gw.domain.core.content.atlas;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class AtlasCore {

  private Long id;

  private Long userId;

  private Instant createAt;

  private Instant updateAt;

  private Integer status;


  public AtlasCore setId(Long id) {
    this.id = id;
    return this;
  }

  public AtlasCore setUserId(Long userId) {
    this.userId = userId;
    return this;
  }

  public AtlasCore setCreateAt(Instant createAt) {
    this.createAt = createAt;
    return this;
  }

  public AtlasCore setUpdateAt(Instant updateAt) {
    this.updateAt = updateAt;
    return this;
  }

  public AtlasCore setStatus(Integer status) {
    this.status = status;
    return this;
  }
}
