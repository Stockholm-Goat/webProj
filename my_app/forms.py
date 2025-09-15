from django import forms

# not used

class raw_input_name(forms.Form):
    name = forms.CharField(max_length=255)


class raw_input_df(forms.Form):
    file = forms.FileField()


class ParamForm(forms.Form):
    param1 = forms.CharField()
    param2 = forms.IntegerField()
