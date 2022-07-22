package com.duitang.dm.learning.ddd.gw.port.web.dto.req;

import com.duitang.dm.learning.ddd.gw.infra.core.BaseSearchPage;
import java.io.Serializable;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class SearchReq implements Serializable {

  private static final long serialVersionUID = 2721785723520903980L;

  private BaseSearchPage page;

  private String searchTxt;

  public SearchReq setPage(BaseSearchPage page) {
    this.page = page;
    return this;
  }
}
