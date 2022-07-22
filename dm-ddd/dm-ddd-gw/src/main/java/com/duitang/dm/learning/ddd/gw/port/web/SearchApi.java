package com.duitang.dm.learning.ddd.gw.port.web;

import com.duitang.dm.learning.ddd.gw.application.AtlasSearchApp;
import com.duitang.dm.learning.ddd.gw.application.dto.UserSearchAtlasResp;
import com.duitang.dm.learning.ddd.gw.port.web.dto.req.SearchReq;

public class SearchApi {

  private AtlasSearchApp atlasSearchApp;


  public void searchAtlas(SearchReq req) {
    /*1. 处理请求*/
    /*2. 调用应用服务*/
    UserSearchAtlasResp resp = atlasSearchApp.userSearchAtlas(req, 0L);
    /*3. 处理响应, 封装成 vo*/
  }

}
