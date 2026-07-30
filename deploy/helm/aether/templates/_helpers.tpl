{{/* Expand the name of the chart. */}}
{{- define "aether.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name. Truncated to 63 chars because some Kubernetes name
fields are limited to that by the DNS spec.
*/}}
{{- define "aether.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "aether.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "aether.labels" -}}
helm.sh/chart: {{ include "aether.chart" . }}
{{ include "aether.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: aether
{{- end }}

{{- define "aether.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aether.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: inference
{{- end }}

{{/* Image reference; tag falls back to the chart's appVersion. */}}
{{- define "aether.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
