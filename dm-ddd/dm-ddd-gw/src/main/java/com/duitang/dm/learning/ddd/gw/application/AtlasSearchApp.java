package com.duitang.dm.learning.ddd.gw.application;

import com.duitang.dm.learning.ddd.gw.application.dto.UserSearchAtlasResp;
import com.duitang.dm.learning.ddd.gw.domain.assemble.AtlasTinyAssemble;
import com.duitang.dm.learning.ddd.gw.domain.service.AtlasSrv;
import com.duitang.dm.learning.ddd.gw.port.web.dto.req.SearchReq;
import java.util.List;
import java.util.Map;

public class AtlasSearchApp {

  private AtlasSrv atlasSrv;

  public UserSearchAtlasResp userSearchAtlas(SearchReq req, Long currentUser) {
    /*1. 通过搜索系统去查询所有的 atlas*/
    List<Long> atlasIds = doSearch();

    /*2. 组装出 atlas 常用的基础聚合信息*/
    List<AtlasTinyAssemble> atlasTinyAssembles = atlasSrv.queryTinyAtlasAggs(atlasIds);

    /*3. 查询当前用户对每个 列表的互动情况. 比如是否点赞过, 是否收藏过信息*/

    /*fixme, 下面 用户 action 行为由于比较通用, 可以封装在 atlasSrv 中为多个 appSrv 中服务 */
    Map<Long, Boolean> atlasLikeMap = queryAtlasLikeMap(atlasIds, currentUser);
    Map<Long, Boolean> atlasFavMap = queryAtlasFavMap(atlasIds, currentUser);

    /*4. */
    return new UserSearchAtlasResp(
        atlasTinyAssembles,
        atlasLikeMap,
        atlasFavMap
    );
  }

  private Map<Long, Boolean> queryAtlasFavMap(List<Long> atlasIds, Long currentUser) {
    return null;
  }

  private Map<Long, Boolean> queryAtlasLikeMap(List<Long> atlasIds, Long currentUser) {
    return null;
  }

  private List<Long> doSearch() {
    return null;
  }
}
