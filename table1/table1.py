"""
    table1.py

    The main Table1 module, containing the Table1 class
"""

__author__ = 'Christian Haudenschild'


# TODO: Add kruskal-wallis for median[IQR]


class Table1():
    '''
    Creates table 1 from pandas Dataframe and allows for export in docx, xls format

    Attributes
    ----------
    table : pandas Dataframe
        output table

    '''
    plus_minus = u'\u00B1'

    def __init__(self, df, stratification_var, names, keep_one_vars=None,  rownames=None, colnames=None, col_ordering=None, row_ordering=None, rounding_digits=1, include_overall=True, overall_colname='Overall', total_row=True, deviation_measure_for_numeric='sd', p_val=True):
        '''
        Parameters
        ----------
        :param df : pd.DataFrame
            Input dataframe
        :param stratification_var : str
            Column stratification variable
        :param names : Dict[str, str]
            Specifies variables that are to be in the table based on keys. Values contain name mapping for baseline variables to new names. Also  All following parameters and methods use the new names as reference.
        :param keep_one_vars : Dict[str, list], optional
            In the case of multilevel variables, allows one to pass name:level items such that within variable name, only the given level is retained as a single row (default:None)
        :param rownames : Dict[str, str], optional
            Specify rownames with format old name: new name (default:None)
        :param colnames : Dict[str, str], optional
            Specify colnames with format old name: new name (default:None)
        :param col_ordering : list, optional
            Order of the stratification variable (default:None)
        :param row_ordering : Dict[str, list], optional
            Pass name:order items such that variable name is ordered according to order (default:None)
        :param rouding_digits : int, optional
            Number of digits to round data to (default:2)
        :param include_overall : bool, optional
            Inserts a row-wise total column (default:True)
        :param overall_colname: str, optional
            Name of total column (default:'Overall')
        :param total_row: bool, optional
            Inserts a row with column-wise totals at top of table (default:True)
        :param deviation_measure_for_numeric: 'str'
            For numeric variables, select deviation measure - either 'sd' for standard deviation or 'se' for standard error of the mean (default:'sd')
        :param p_val: bool
            Calculate Pearson’s chi-squared test for independence for categorical data or one-way analysis of variance for continuous data and add p_value in a column
        '''
        assert deviation_measure_for_numeric in ['se', 'sd']
        if colnames:
            assert isinstance(colnames, dict)
        if row_ordering:
            assert isinstance(row_ordering, dict)

        self.df = df
        self.stratification_var = stratification_var
        self.names = names
        self.keep_one_vars = keep_one_vars
        self.rownames = rownames
        self.colnames = colnames
        self.col_ordering = col_ordering
        self.row_ordering = row_ordering
        self.rounding_digits = rounding_digits
        self.include_overall = include_overall
        self.overall_colname = overall_colname
        self.total_row = total_row
        self.deviation_measure_for_numeric = deviation_measure_for_numeric
        self.p_val = p_val
        self.reverse_names = {v: k for k, v in self.names.items()}

        self.make_table()

    @staticmethod
    def create(*args, **kwargs):
        return Table1(*args, **kwargs).table

    def _is_categorical(self, var):
        return self.df[var].dtype.name in ['bool', 'object', 'category']

    def _p_categorical(self, group):
        if self.include_overall:
            stat, p, dof, expected = scipy.stats.chi2_contingency(
                group.drop(columns=self.overall_colname))
        else:
            stat, p, dof, expected = scipy.stats.chi2_contingency(group)

        return '<0.001' if p < 0.001 else round(p, 3)

    def _p_anova(self, var):

        unique_levels = [
            x for x in self.df[self.stratification_var].unique() if pd.notna(x)]

        measures = [self.df.loc[self.df[self.stratification_var]
                                == level, var] for level in unique_levels]

        for i, m in enumerate(measures):
            measures[i] = [x for x in m if pd.notna(x)]

        stat, p = scipy.stats.f_oneway(*measures)

        return '<0.001' if p < 0.001 else round(p, 3)

    def _make_categorical_minitable(self, var):

        def reformat_categorical(col):
            percs = col / group_col_totals[col.name] * 100
            percs = percs.round(self.rounding_digits)
            return pd.Series([f'{val} ({perc})' for val, perc in zip(col, percs)], index=col.index, name=col.name)

        group = self.df.groupby(self.stratification_var)[
            var].value_counts(dropna=False).rename().reset_index()
        group = group.pivot(var, self.stratification_var)
        group.columns = group.columns.droplevel(0)
        group.columns.name = ''
        group.rename({np.nan: f'Missing {self.names[var]}'}, inplace=True)
        group.index.name = self.names[var]

        if self.include_overall:
            group[self.overall_colname] = group.sum(axis=1)

        group_col_totals = group.sum(axis=0)
        group = group.fillna(0).astype(int)

        if self.p_val:
            p = self._p_categorical(group)

        group = group.apply(reformat_categorical, axis=0)

        if self.p_val:
            p_series = pd.Series(
                [p] + ['' for _ in range(1, len(group))], index=group.index, name='p-value')

        if self.keep_one_vars and self.names[var] in self.keep_one_vars.keys():
            to_drop = []
            for i in group.index:
                if i in self.rownames.keys():
                    if self.rownames[i] != self.keep_one_vars[self.names[var]]:
                        to_drop.append(i)
                else:
                    if i != self.keep_one_vars[self.names[var]]:
                        to_drop.append(i)

            group = group.drop(index=to_drop)
            group.rename(
                {self.keep_one_vars[self.names[var]]: group.index.name}, inplace=True)

            if self.p_val:
                group['p-value'] = p

        else:
            header = pd.DataFrame(pd.Series(['' for _ in range(
                len(group.columns))], index=group.columns, name=self.names[var])).transpose()

            group = pd.concat([header, group])

            if self.p_val:
                p_series = pd.Series(
                    [p] + ['' for _ in range(1, len(group))], index=group.index, name='p-value')
                group = pd.concat([group, p_series], axis=1)

        return group

    def _make_numeric_row(self, var):

        overall_mn = self.df[var].mean()
        mns = self.df.groupby(self.stratification_var)[var].mean()

        if self.deviation_measure_for_numeric == 'se':
            overall_dev = self.df[var].std() / np.sqrt(len(self.df[var]))
            devs = self.df.groupby(self.stratification_var)[var].apply(
                lambda x: x.std() / np.sqrt(len(x)))
        elif self.deviation_measure_for_numeric == 'sd':
            overall_dev = self.df[var].std()
            devs = self.df.groupby(self.stratification_var)[var].apply(np.std)

        if self.include_overall:
            mns = mns.append(
                pd.Series(overall_mn, index=[self.overall_colname]))
            devs = devs.append(
                pd.Series(overall_dev, index=[self.overall_colname]))

        overall_mn = round(overall_mn, self.rounding_digits)
        mns = mns.round(self.rounding_digits)
        devs = devs.round(self.rounding_digits)

        ser = pd.DataFrame(pd.Series([f'{mn} {self.plus_minus} {sd}' for mn, sd in zip(
            mns, devs)], index=mns.index, name=self.names[var])).transpose()

        if self.p_val:
            p = self._p_anova(var)
            ser['p-value'] = p

        return ser

    def make_table(self):

        self.table = pd.concat([self._make_categorical_minitable(var) if self._is_categorical(
            var) else self._make_numeric_row(var) for var in self.names.keys()])

        if self.total_row:
            self.insert_total_row(return_table=False)

        if self.colnames:
            self.table = self.table.rename(columns=self.colnames)

        if self.row_ordering:
            for var, order in self.row_ordering.items():
                self.row_reorder(var, order, return_table=False)

        if self.rownames:
            self.table.rename(self.rownames, inplace=True)

        if self.col_ordering:
            assert len(self.col_ordering) == len(
                self.table.columns), f'Got {len(self.col_ordering)} in col_ordering, expected {len(self.table.columns)}'
            self.column_reorder(self.col_ordering)

    def column_reorder(self, order, return_table=True):

        assert all([o in self.table.columns for o in order])

        table = self.table[order]

        self.table = table

        if return_table:
            return self.table

    def row_reorder(self, var, order, return_table=True):

        og_varname = self.reverse_names[var]

        assert self._is_categorical(og_varname)
        assert var in self.table.index

        try:
            assert len(order) == len(self.df[og_varname].unique())
        except AssertionError:
            unique_levels = self.df[og_varname].unique()
            np.place(unique_levels, pd.isna(unique_levels), f'Missing {var}')
            discrepancies = set(unique_levels).difference(set(order))
            raise ValueError(
                f'{discrepancies} found in levels, not provided in order')

        i_order = [self.table.index.get_loc(o) for o in order]
        new_order = list(np.arange(min(i_order))) + i_order + \
            list(np.arange(max(i_order) + 1, len(self.table)))

        self.table = self.table.iloc[new_order]

        if return_table:
            return self.table

    def insert_header(self, name, after):
        header = pd.DataFrame(pd.Series(['' for _ in range(
            len(self.table.columns))], index=self.table.columns, name=name)).transpose()

        idx = self.table.index.get_loc(after)
        self.table = pd.concat(
            [self.table.iloc[:idx + 1], header, self.table.iloc[idx + 1:]])

        return self.table

    def insert_total_row(self, adornment='n = ', return_table=True):

        counts = self.df[self.stratification_var].value_counts(dropna=False)
        counts[self.overall_colname] = len(self.df)
        counts = pd.Series(
            [f'{adornment}{c}' for c in counts], index=counts.index, name='')
        sum_row = pd.DataFrame(counts).transpose()

        if self.p_val:
            sum_row['p-value'] = ''

        sum_row = sum_row[list(self.table.columns)]
        self.table = pd.concat([sum_row, self.table])

        if return_table:
            return self.table

    def to_excel(self, fname):

        df = self.table.reset_index().rename(columns={'index': ''})
        df.to_excel(fname)

    def to_word(self, fname, style=None):

        df = self.table.reset_index().rename(columns={'index': ''})
        doc = docx.Document()

        t = doc.add_table(df.shape[0] + 1, df.shape[1])
        for j in range(df.shape[1]):
            t.cell(0, j).text = df.columns[j]
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                t.cell(i + 1, j).text = str(df.values[i, j])
        t.style = 'Table Grid' if style is None else style

        doc.save(fname)
