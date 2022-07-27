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
    Mustache m = factory.compile(
        "/Users/carl/IdeaProjects/dm-learning/code-maker/src/main/resources/tpl/mybatis/mapper_xml.mustache");

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
    List<RenderColumn> pksOutput = pks.stream().map(x -> toOutputColumn(x, cfg)).collect(Collectors.toList());
    renderContext.setPks(pksOutput);

    List<RenderColumn> colsOutput = columns.stream().map(x -> toOutputColumn(x, cfg)).collect(Collectors.toList());
    renderContext.setCols(colsOutput);

    /*2. render mybatis all cols*/
    List<String> allCols = new ArrayList<>(columns.size() + pks.size());
    allCols.addAll(pks.stream().map(MysqlColumn::getColumnName).collect(Collectors.toList()));
    allCols.addAll(columns.stream().map(MysqlColumn::getColumnName).collect(Collectors.toList()));
    renderContext.setAllCols(StringTools.JOINER.join(allCols));




    /*3. render find by pk*/

    String parameterType = "java.util.Map";
    if (pkSize == 1) {
      parameterType = pks.get(0).getJavaTypeMapping().getJavaClass().getName();
    }

    FindByPk findByPk = new FindByPk().setParameterType(parameterType).setTableName(cfg.getTableName())
        .setWhereSql(Joiner.on(" AND ")
                         .join(pksOutput.stream().map(x -> String.format("%s=#{%s}", x.getColName(), x.getJavaName()))
                                   .collect(Collectors.toList())));

    renderContext.setFindByPk(findByPk);

    /*4. render update by version*/
    if (cfg.getVersionColName() != null) {
      List<RenderColumn> otherColumns = new ArrayList<>();
      RenderColumn versionCol = null;

      for (RenderColumn renderColumn : colsOutput) {
        if (Objects.equals(renderColumn.getColName(), cfg.getVersionColName())) {
          versionCol = renderColumn;
        } else {
          otherColumns.add(renderColumn);
        }
      }

      if (versionCol != null) {
        List<RenderColumn> whereCols = new ArrayList<>();
        whereCols.addAll(pksOutput);
        whereCols.add(versionCol);

        String whereSql = Joiner.on(" AND ")
            .join(whereCols.stream().map(x -> String.format("%s=#{%s}", x.getColName(), x.getJavaName()))
                      .collect(Collectors.toList()));
        renderContext.setUpdateByVersion(new UpdateByVersion().setCols(otherColumns).setVersion(versionCol)
                                             .setWhereSql(whereSql).setTableName(cfg.getTableName()));


      } else {
        log.warn("can't find version col , colName:{}, tableName:{}", cfg.getVersionColName(), cfg.getTableName());
      }


      /*5. render insertOne*/
      List<RenderColumn> allColsOutput = new ArrayList<>();
      String useGeneratedKeysStr = "";
      for (MysqlColumn pk : pks) {
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

      allColsOutput.addAll(colsOutput);
      renderContext.setInsertOne(new InsertOne().setClassId(cfg.getDataObjectClass()).setCols(allColsOutput)
                                     .setTableName(cfg.getTableName()).setUseGeneratedKeysStr(useGeneratedKeysStr));
    }

    return renderContext;
  }


  private RenderColumn toOutputColumn(MysqlColumn mysqlColumn, Cfg cfg) {
    DefaultMappingStrategy mappingStrategy = cfg.getMappingStrategy();
    RenderColumn renderColumn = new RenderColumn();
    renderColumn.setJavaName(StringTools.formatConvert(mappingStrategy, mysqlColumn.getColumnName()));
    renderColumn.setColName(mysqlColumn.getColumnName());
    return renderColumn;
  }


}
