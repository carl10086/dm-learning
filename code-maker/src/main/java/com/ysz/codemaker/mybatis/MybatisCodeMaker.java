package com.ysz.codemaker.mybatis;

import com.github.mustachejava.DefaultMustacheFactory;
import com.github.mustachejava.Mustache;
import com.github.mustachejava.MustacheFactory;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.ysz.codemaker.mybatis.core.render.items.InsertOne;
import com.ysz.codemaker.toos.mysql.MysqlMetaQuery;
import com.ysz.codemaker.toos.mysql.core.MysqlColumn;
import com.ysz.codemaker.toos.mysql.core.MysqlMeta;
import com.ysz.codemaker.toos.tools.StringTools;
import com.ysz.codemaker.mybatis.core.Cfg;
import com.ysz.codemaker.mybatis.core.DefaultMappingStrategy;
import com.ysz.codemaker.mybatis.core.Output;
import com.ysz.codemaker.mybatis.core.render.RenderContext;
import com.ysz.codemaker.mybatis.core.render.RenderColumn;
import com.ysz.codemaker.mybatis.core.render.items.FindByPk;
import com.ysz.codemaker.mybatis.core.render.items.UpdateByVersion;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class MybatisCodeMaker {

  MustacheFactory factory = new DefaultMustacheFactory();

  public Output execute(Cfg cfg) throws Exception {
    final MysqlMeta mysqlMeta = new MysqlMetaQuery().queryColumns(cfg.getMysql(),
                                                                  cfg.getDatabase(),
                                                                  cfg.getTableName()
    );

    Output output = new Output().setMapperXml(renderMybatisMapperXml(mysqlMeta, cfg));

    return output;
  }

  private String renderMybatisMapperXml(MysqlMeta mysqlMeta, Cfg cfg) throws Exception {
    Mustache m = factory.compile(cfg.getMapperXmlMustache());

    StringWriter stringWriter = new StringWriter();

    RenderContext renderContext = render(mysqlMeta, cfg);
    m.execute(stringWriter, Lists.newArrayList(renderContext));

    return stringWriter.toString();
  }


  private RenderContext render(MysqlMeta mysqlMeta, Cfg cfg) {
    RenderContext renderContext = new RenderContext();
    final List<MysqlColumn> pks = mysqlMeta.getPks();
    final List<MysqlColumn> columns = mysqlMeta.getColumns();
    int pkSize = pks.size();

    /*1. render mybatis result map*/
    renderContext.setAllCols(allCols(cfg, mysqlMeta));


    /*2. render find by pk*/

    renderContext.setFindByPk(new FindByPk().setParameterType(parameterType(pks)).setTableName(cfg.getTableName())
                                  .setWhereSql(findByPkWhereSql(cfg, pks)));

    /*3. render update by version*/
    if (cfg.getVersionColName() != null) {
      UpdateByVersion updateByVersion = updateByVersion(cfg, mysqlMeta);
      if (updateByVersion != null) {
        renderContext.setUpdateByVersion(updateByVersion);
      }
    }


    /*4. 渲染 insertOne*/
    renderContext.setInsertOne(insertOne(cfg, mysqlMeta));

    return renderContext;
  }

  private InsertOne insertOne(Cfg cfg, MysqlMeta mysqlMeta) {
    List<RenderColumn> allColsOutput = new ArrayList<>();
    String useGeneratedKeysStr = "";
    for (MysqlColumn pk : mysqlMeta.getPks()) {
      if (pk.auto()) {
        RenderColumn renderColumn = toOutputColumn(pk, cfg);
        useGeneratedKeysStr = String.format("keyProperty=\"%s\" keyColumn=\"%s\" useGeneratedKeys=\"true\"",
                                            renderColumn.getJavaName(),
                                            renderColumn.getColName()
        );
      } else {
        allColsOutput.add(toOutputColumn(pk, cfg));
      }
    }

    allColsOutput.addAll(mysqlMeta.getColumns().stream().map(x1 -> toOutputColumn(x1, cfg))
                             .collect(Collectors.toList()));
    InsertOne insertOne = new InsertOne().setClassId(cfg.getDataObjectClass()).setCols(allColsOutput)
        .setTableName(cfg.getTableName()).setUseGeneratedKeysStr(useGeneratedKeysStr);
    return insertOne;
  }

  private UpdateByVersion updateByVersion(Cfg cfg, MysqlMeta mysqlMeta) {
    List<MysqlColumn> columns = mysqlMeta.getColumns();
    List<MysqlColumn> pks = mysqlMeta.getPks();
    UpdateByVersion updateByVersion = null;

    /*要区分 version 列和 非 version 列, version 的 set 是 version ++ , 其他列是 = */
    List<RenderColumn> otherColumns = new ArrayList<>();
    RenderColumn versionCol = null;

    for (RenderColumn renderColumn : columns.stream().map(x1 -> toOutputColumn(x1, cfg)).collect(Collectors.toList())) {
      if (Objects.equals(renderColumn.getColName(), cfg.getVersionColName())) {
        versionCol = renderColumn;
      } else {
        otherColumns.add(renderColumn);
      }
    }

    /*找到了 version 列*/
    if (versionCol != null) {
      List<RenderColumn> whereCols = new ArrayList<>();
      whereCols.addAll(pks.stream().map(x1 -> toOutputColumn(x1, cfg)).collect(Collectors.toList()));
      whereCols.add(versionCol);

      String whereSql = Joiner.on(" AND ")
          .join(whereCols.stream().map(x -> String.format("%s=#{%s}", x.getColName(), x.getJavaName()))
                    .collect(Collectors.toList()));
      updateByVersion = new UpdateByVersion().setCols(otherColumns).setVersion(versionCol).setWhereSql(whereSql)
          .setTableName(cfg.getTableName());


    } else {
      log.warn("can't find version col , colName:{}, tableName:{}", cfg.getVersionColName(), cfg.getTableName());
    }
    return updateByVersion;
  }

  private String findByPkWhereSql(Cfg cfg, List<MysqlColumn> pks) {
    return Joiner.on(" AND ")
        .join(pks.stream().map(x11 -> toOutputColumn(x11, cfg)).collect(Collectors.toList()).stream()
                  .map(x -> String.format("%s=#{%s}", x.getColName(), x.getJavaName())).collect(Collectors.toList()));
  }

  private String parameterType(List<MysqlColumn> pks) {
    String parameterType = "java.util.Map";
    if (pks.size() == 1) {
      parameterType = pks.get(0).getJavaTypeMapping().getJavaClass().getName();
    }
    return parameterType;
  }

  private List<RenderColumn> allCols(
      Cfg cfg, MysqlMeta mysqlMeta
  ) {
    List<MysqlColumn> pks = mysqlMeta.getPks();
    List<MysqlColumn> columns = mysqlMeta.getColumns();
    final List<RenderColumn> allCols = new ArrayList<>(pks.size() + columns.size());
    allCols.addAll(pks.stream().map(x11 -> toOutputColumn(x11, cfg)).collect(Collectors.toList()));
    allCols.addAll(columns.stream().map(x1 -> toOutputColumn(x1, cfg)).collect(Collectors.toList()));
    return allCols;
  }


  private RenderColumn toOutputColumn(MysqlColumn mysqlColumn, Cfg cfg) {
    DefaultMappingStrategy mappingStrategy = cfg.getMappingStrategy();
    RenderColumn renderColumn = new RenderColumn();
    renderColumn.setJavaName(StringTools.formatConvert(mappingStrategy, mysqlColumn.getColumnName()));
    renderColumn.setColName(mysqlColumn.getColumnName());
    return renderColumn;
  }


}
